import random

import numpy as np
from PIL import Image
from torchvision import transforms as T

img1 = np.load('dataset/train/masks/IMG-0002-00056-crop.npy')
img = Image.fromarray(img1).convert('L')

aspect_ratio = img.size[1] / img.size[0]

Transform = []

ResizeRange = random.randint(300, 320)
Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
p_transform = random.random()

Transform.append(T.RandomRotation((0, 20)))

RotationRange = random.randint(-10, 10)
Transform.append(T.RandomRotation((RotationRange, RotationRange)))
CropRange = random.randint(250, 270)
Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
Transform = T.Compose(Transform)

image = Transform(img)

print(img1)
