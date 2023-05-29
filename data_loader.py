import os
import random

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=224, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        # """
        # 初始化图像路径和预处理模块。
        #
        # 参数:
        #     root (str): 数据集根目录路径。
        #     image_size (int): 图像尺寸，默认为224。
        #     mode (str): 数据集模式，默认为'train'。
        #     augmentation_prob (float): 数据增强的概率，默认为0.4。
        # """
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root + 'masks'
        self.image_paths = list(map(lambda x: os.path.join(root+'imgs', x), os.listdir(root+'imgs')))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        # """
        #         从文件中读取图像并进行预处理，然后返回图像及其标签。
        #
        #         参数:
        #             index (int): 数据集索引。
        #
        #         返回:
        #             image (Tensor): 预处理后的图像张量。
        #             GT (Tensor): 预处理后的标签张量。
        #             这部分代码是数据集类ImageDataset中的__getitem__方法，
        #             用于获取数据集中指定索引的数据项。
        #             首先，通过索引获取图像路径image_path和对应的标签路径GT_path。
        #             然后，使用Image.open方法打开图像和标签，并将图像转换为RGB模式，标签转换为灰度模式。
        #             接下来，计算图像的宽高比aspect_ratio。定义一个空列表Transform用于存储数据增强操作。
        #             首先，生成一个随机的ResizeRange，然后将图像的高度根据宽高比进行调整，保持宽度不变。
        #             然后，根据设定的数据增强概率self.augmentation_prob决定是否进行数据增强操作。
        #             如果小于等于概率值，则进行数据增强。随机选择旋转角度rotation_degree，并将其加入到Transform中。
        #             随机进行水平翻转和垂直翻转操作。然后，使用T.Compose方法将所有的数据增强操作组合起来，并将其应用于图像和标签。
        #             定义另一个空列表Transform用于存储图像的最终处理操作。
        #             将图像调整为指定的尺寸self.image_size，然后将图像和标签转换为张量类型。最后，返回预处理后的图像image和标签GT。
        #         """
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        # 图像路径

        filename = image_path.split('/')[-1]
        # 从图像路径中提取文件名

        GT_path = self.GT_paths + '/' + filename
        # 构建标签的完整路径

        image = Image.open(image_path)
        image = image.convert('RGB')
        # 打开图像，并转换为RGB模式

        # image = np.load(image_path, allow_pickle=True)
        # image = Image.fromarray(image.astype('uint8')).convert('RGB')
        # 如果需要，可以加载图像数据并进行转换

        GT = Image.open(GT_path)
        GT = GT.convert('L')
        # 打开标签图像，并转换为灰度模式（L模式）

        # GT = np.load(GT_path, allow_pickle=True)
        # GT = Image.fromarray(GT.astype('uint8')).convert('L')
        # 如果需要，可以加载标签数据并进行转换

        aspect_ratio = image.size[1] / image.size[0]
        # 计算图像的宽高比

        Transform = []
        # '''
        # 这段代码是数据增强的部分。首先，根据一定范围随机调整图像大小，保持宽高比。然后，生成一个随机数p_transform，用于判断是否进行数据增强操作。
        #
        #     如果当前模式为训练模式且满足数据增强的概率条件，将进行以下操作：
        #
        #     随机选择旋转角度并根据角度调整宽高比。
        #     随机对图像进行旋转。
        #     进一步随机旋转图像。
        #     随机中心裁剪图像。
        #     将变换操作应用于图像和标签。
        #     随机裁剪图像的一部分。
        #     这样，就完成了数据增强的过程。
        # '''

        ResizeRange = random.randint(300, 320)  # 随机生成图像缩放范围，取值在300到320之间
        Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))  # 将图像按照生成的缩放范围进行调整，保持宽高比

        p_transform = random.random()  # 生成一个随机概率值

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            # 如果当前模式是训练模式且随机概率小于等于数据增强的概率，则进行数据增强操作
            RotationDegree = random.randint(0, 3)  # 随机选择旋转角度，取值为0、1、2、3
            RotationDegree = self.RotationDegree[RotationDegree]  # 获取对应的旋转角度值
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1 / aspect_ratio  # 根据旋转角度调整宽高比

            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))  # 对图像进行随机旋转

            RotationRange = random.randint(-10, 10)  # 随机选择旋转范围，取值在-10到10之间
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))  # 对图像进行进一步的随机旋转
            CropRange = random.randint(250, 270)  # 随机选择裁剪范围，取值在250到270之间
            Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))  # 对图像进行中心裁剪
            Transform = T.Compose(Transform)  # 将所有数据增强操作组合成一个变换

            image = Transform(image)  # 对图像进行数据增强变换
            GT = Transform(GT)  # 对Ground Truth进行相同的数据增强变换

            ShiftRange_left = random.randint(0, 20)  # 随机选择左偏移范围，取值在0到20之间
            ShiftRange_upper = random.randint(0, 20)  # 随机选择上偏移范围，取值在0到20之间
            ShiftRange_right = image.size[0] - random.randint(0, 20)  # 随机选择右偏移范围，取值在0到20之间
            ShiftRange_lower = image.size[1] - random.randint(0, 20)  # 随机选择下偏移范围，取值在0到20之间
            image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))  # 对图像进行裁剪
            GT = GT.crop(
                box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))  # 对Ground Truth进行相同的裁剪

            if random.random() < 0.5:
                image = F.hflip(image) # 随机水平翻转图像
                GT = F.hflip(GT)# 对Ground Truth进行相同的水平翻转

            if random.random() < 0.5:
                image = F.vflip(image)  # 随机垂直翻转图像
                GT = F.vflip(GT)  # 对Ground Truth进行相同的垂直翻转

            Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)  # 随机调整图像颜色的亮度、对比度和色调

            image = Transform(image)  # 对图像进行颜色调整

            Transform = []  # 清空变换列表
        Transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)))  # 将图像调整为指定大小
        Transform.append(T.ToTensor())  # 将图像转换为Tensor
        Transform = T.Compose(Transform)  # 将所有预处理操作组合成一个变换

        image = Transform(image)  # 对图像进行预处理变换
        GT = Transform(GT)  # 对Ground Truth进行相同的预处理变换

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化图像
        image = Norm_(image)  # 对图像进行标准化处理

        return image, GT  # 返回处理后的图像和Ground Truth

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
    # 在函数内部，首先创建了一个
    # ImageFolder
    # 数据集对象，该对象用于加载图像数据，并可以进行数据增强操作。然后，使用
    # data.DataLoader
    # 创建了数据加载器，设置了批量大小、是否打乱数据以及并行加载的线程数。最后，将数据加载器返回。
    """构建并返回数据加载器。"""

    # 创建数据集对象
    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)

    # 创建数据加载器
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

