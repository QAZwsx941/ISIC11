import argparse
import os
import random

from torch.backends import cudnn

from data_loader import get_loader
from solver import Solver
#这段代码主要是一个PyTorch深度学习项目的主函数，
#主要负责初始化一些参数，创建文件夹，加载数据，构建模型，训练和测试模型等。


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

    #定义主函数，传入的参数为config，cudnn.benchmark设置为True以加速计算，
    # if语句检查config.model_type是否在[
    #'U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']
    #中，如果不在其中，则输出错误信息并返回。

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    #生成一些随机参数，包括学习率lr，
    # 数据增强概率augmentation_prob，
    # 训练轮数epoch，
    # 学习率下降比例decay_ratio
    # 和学习率下降轮数decay_epoch

    lr = random.random() * 0.0005 + 0.0000005
    augmentation_prob = random.random() * 0.7
    epoch = random.choice([100, 150, 200, 250])
    decay_ratio = random.random() * 0.8
    decay_epoch = int(epoch * decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
    #获取训练集、验证集和测试集的数据加载器。

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=128)
    #图像的尺寸
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    #R2U_Net或R2AttU_Net模型的循环步数t
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    #输入图像的通道数
    parser.add_argument('--output_ch', type=int, default=1)
    #输出图像的通道数
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=170)
    #学习率开始衰减的迭代次数
    parser.add_argument('--batch_size', type=int, default=1)
    #批量训练的样本数量
    parser.add_argument('--num_workers', type=int, default=8)
    #用于数据加载的线程数
    parser.add_argument('--lr', type=float, default=0.0002)
    #初始学习率
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    #Adam优化器中的动量参数
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    #数据增强的概率

    parser.add_argument('--log_step', type=int, default=2)
    #每隔多少个训练步骤打印一次日志
    parser.add_argument('--val_step', type=int, default=2)
    #每隔多少个训练步骤进行一次验证

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='R2AttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset/val/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    config = parser.parse_args()
    main(config)
