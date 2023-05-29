from PIL import Image
import cv2
import os
import random
import numpy as np

import argparse

import torch
from torchvision import transforms as T

from network import R2AttU_Net


def tensor2img(x):
    img = x.detach().numpy()
    img = img*255
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
