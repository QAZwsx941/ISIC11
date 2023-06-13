import csv
import os

import torch.nn.functional as F
from torch import optim

from evaluation import *
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from PIL import Image
import cv2
import os
import random
import numpy as np

import argparse

import torch
from torchvision import transforms as T

from network import R2AttU_Net


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # """
        # 初始化方法，用于创建一个新的实例。
        #
        # 参数：
        # - config: 配置对象，包含模型的各项设置和超参数
        # - train_loader: 训练数据的数据加载器
        # - valid_loader: 验证数据的数据加载器
        # - test_loader: 测试数据的数据加载器
        # """

        # 数据加载器
        self.train_loader = train_loader  # 训练数据加载器
        self.valid_loader = valid_loader  # 验证数据加载器
        self.test_loader = test_loader  # 测试数据加载器

        # 模型
        self.unet = None  # UNet模型实例
        self.optimizer = None  # 优化器实例
        self.img_ch = config.img_ch  # 输入图像的通道数
        self.output_ch = config.output_ch  # 模型输出的通道数
        self.criterion = torch.nn.BCELoss()  # 损失函数，二分类交叉熵
        self.augmentation_prob = config.augmentation_prob  # 数据增强的概率

        # 超参数
        self.lr = config.lr  # 学习率
        self.beta1 = config.beta1  # 优化器的超参数
        self.beta2 = config.beta2  # 优化器的超参数

        # 训练设置
        self.num_epochs = config.num_epochs  # 总训练轮数
        self.num_epochs_decay = config.num_epochs_decay  # 学习率衰减轮数
        self.batch_size = config.batch_size  # 批次大小

        # 步长
        self.log_step = config.log_step  # 日志记录步长
        self.val_step = config.val_step  # 验证步长

        # 路径
        self.model_path = config.model_path  # 模型保存路径
        self.result_path = config.result_path  # 结果保存路径
        self.mode = config.mode  # 模型模式，如训练、验证或测试

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备类型
        self.model_type = config.model_type  # 模型类型
        self.t = config.t  # 当前训练轮数
        self.build_model()  # 构建模型

    def build_model(self):
        # """
        # 构建生成器和判别器模型。
        # """
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=3, output_ch=1)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=3, output_ch=1, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=3, output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=1, t=self.t)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)

    def print_network(self, model, name):
        # """
        # 打印网络的信息。
        #
        # 参数：
        # - model: 要打印的网络模型
        # - name: 模型的名称
        # """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("参数数量：{}".format(num_params))

    def to_data(self, x):
        # """
        # 将变量转换为张量。
        #
        # 参数：
        # - x: 要转换的变量
        #
        # 返回：
        # 转换后的张量数据
        # """
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self):
        """
        更新学习率。
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def reset_grad(self):
        """
        将梯度缓冲区清零。
        """
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        # """
        # 计算准确率。
        #
        # 参数：
        # - SR: 分割结果张量
        # - GT: 真实标签张量
        # """
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        # """
        # 将张量转换为图像。
        #
        # 参数：
        # - x: 输入张量
        #
        # 返回：
        # 转换后的二值图像张量
        # """
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """
        训练编码器、生成器和判别器。
        """
        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # 加载预训练的编码器
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s 从 %s 成功加载' % (self.model_type, unet_path))
        else:
            # 训练编码器
            lr = self.lr
            best_unet_score = 0.

            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss = 0

                acc = 0.  # 准确率
                SE = 0.  # 敏感性（召回率）
                SP = 0.  # 特异性
                PC = 0.  # 精确度
                F1 = 0.  # F1得分
                JS = 0.  # 杰卡德相似度
                DC = 0.  # Dice系数
                length = 0

                for i, (images, GT) in enumerate(self.train_loader):
                    # GT : Ground Truth

                    images = images.to(self.device)  # 16*3*256*256
                    GT = GT.to(self.device)  # 16*1*256*256

                    # SR : Segmentation Result
                    SR = self.unet(images)
                    SR_probs = F.sigmoid(SR)
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)  # shape: B*65536

                    GT_flat = GT.view(GT.size(0), -1)  # shape:B*65536
                    loss = self.criterion(SR_flat, GT_flat)
                    epoch_loss += loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    acc += get_accuracy(SR, GT)  # 准确率
                    SE += get_sensitivity(SR, GT)  # 敏感性（召回率）
                    SP += get_specificity(SR, GT)  # 特异性
                    PC += get_precision(SR, GT)  # 精确度
                    F1 += get_F1(SR, GT)  # F1得分
                    JS += get_JS(SR, GT)  # 杰卡德相似度
                    DC += get_DC(SR, GT)  # Dice系数
                    length += images.size(0)

                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length

                # 打印日志信息
                print(
                    'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                        epoch + 1, self.num_epochs, \
                        epoch_loss, \
                        acc, SE, SP, PC, F1, JS, DC))

                # 衰减学习率
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('将学习率衰减为 lr: {}.'.format(lr))

                # ===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0
                for i, (images, GT) in enumerate(self.valid_loader):
                    images = images.to(self.device)
                    # 将图像数据移动到设备上（GPU）
                    GT = GT.to(self.device)
                    # 将真实标签移动到设备上（GPU）
                    SR = F.sigmoid(self.unet(images))
                    # 使用 U-Net 模型对图像进行前向传播，并应用 sigmoid 函数将输出转化为概率值
                    acc += get_accuracy(SR, GT)  # 计算准确率
                    SE += get_sensitivity(SR, GT)  # 计算召回率
                    SP += get_specificity(SR, GT)  # 计算特异度
                    PC += get_precision(SR, GT)  # 计算精度
                    F1 += get_F1(SR, GT)  # 计算F1分数
                    JS += get_JS(SR, GT)  # 计算Jaccard相似度
                    DC += get_DC(SR, GT)  # 计算Dice系数

                    length += images.size(0)

                acc = acc / length  # 平均准确率
                SE = SE / length  # 平均召回率
                SP = SP / length  # 平均特异度
                PC = PC / length  # 平均精度
                F1 = F1 / length  # 平均F1分数
                JS = JS / length  # 平均Jaccard相似度
                DC = DC / length  # 平均Dice系数
                unet_score = JS + DC  # U-Net模型的综合得分

                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                    acc, SE, SP, PC, F1, JS, DC))

                '''
                # 保存验证结果图像（注释掉的代码）
                torchvision.utils.save_image(images.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
                '''

                # 保存最佳的U-Net模型
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                    torch.save(best_unet, unet_path)

            # ===================================== Test ====================================#

            # 删除现有的 U-Net 模型实例，重新构建模型
            del self.unet
            del best_unet
            self.build_model()

            # 加载训练过程中表现最好的 U-Net 模型
            self.unet.load_state_dict(torch.load(unet_path))

            # 将模型设置为评估模式
            self.unet.train(False)
            self.unet.eval()

            acc = 0.  # 准确率
            SE = 0.  # 敏感性（召回率）
            SP = 0.  # 特异性
            PC = 0.  # 精确率
            F1 = 0.  # F1 分数
            JS = 0.  # Jaccard 相似度
            DC = 0.  # Dice 系数
            length = 0

            # 在测试集上进行评估
            for i, (images, GT) in enumerate(self.valid_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = F.sigmoid(self.unet(images))

                # 计算各项评估指标
                acc += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)

                length += images.size(0)

            # 计算平均指标
            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length
            unet_score = JS + DC
            result_file_path = os.path.join(self.result_path, 'result.csv')
            print(result_file_path)
            # 将结果写入结果文件
            f = open(result_file_path, 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow([self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, best_epoch, self.num_epochs,
                         self.num_epochs_decay, self.augmentation_prob])
            f.close()


    def tensor2img(x):
        img = x.detach().numpy()
        img = img * 255
        img = img.astype('uint8')
        return img


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
    # model hyper-parameters
        parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
        parser.add_argument('--img_ch', type=int, default=3)
        parser.add_argument('--output_ch', type=int, default=1)

    # data path
        parser.add_argument('--dir_path', type=str, default='./dataset/test/imgs/')
        parser.add_argument('--save_path', type=str, default='./dataset/test/masks_rgb/')
        parser.add_argument('--model_path', type=str, default='./models/R2AttU_Net-50-0.0001-38-0.4312.pkl')

        config = parser.parse_args()

    # 1. 导入模型
        model = R2AttU_Net(img_ch=config.img_ch, output_ch=config.output_ch, t=config.t)
        model.load_state_dict(torch.load(config.model_path, map_location='cpu'))
        model.eval()

    # 2. 导入图像
        img_files = os.listdir(config.dir_path)
        for file in img_files:
            image = Image.open(os.path.join(config.dir_path, file))
            image = image.convert('RGB')

        aspect_ratio = image.size[1] / image.size[0]

        Transform = []

        ResizeRange = random.randint(300, 320)
        Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
        p_transform = random.random()

        Transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)

        input = image.unsqueeze(dim=0)
        output = torch.sigmoid(model(input)).squeeze()

        out_image = tensor2img(output)
        out_image = np.where(out_image >= 128, 255, 0)

        # out_image = np.where(out_image >= 35, 1, 0)
        # out_image = out_image*255
        out_pil = Image.fromarray(out_image)
        # out_pil.save('result.png')

        out_image = cv2.cvtColor(out_image, cv2.COLOR_GRAY2BGR)
        output_path = os.path.join(config.save_path, file)
        cv2.imwrite(output_path, out_image)
        print(file)
        # cv2.waitKey(0)

        print('done')
