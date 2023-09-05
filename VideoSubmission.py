import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data as tud
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import resnet

resnet56 = resnet.resnet56()
batch_size = 128

midway_model = torch.load(os.path.join("pretrained_models/resnet56-4bfd9763.th"))
resnet56 = torch.nn.DataParallel(resnet56, device_ids=range(torch.cuda.device_count()))
resnet56.load_state_dict(midway_model['state_dict'])
resnet56 = resnet56.to("cuda")
resnet56.eval()

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

normalize = transforms.Normalize(mean=mean,
                                     std=std)

testData = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    normalize,
    ]))

testDataLoader = tud.DataLoader(testData, batch_size=batch_size, shuffle=False, num_workers=2)

# result classes from the  CIFAR10 data set
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# select a random sample for presentation
# random.seed(0)
index = random.sample(range(0, len(testDataLoader.dataset)), 10)
labels = []
images = []

for i in index:
    # image data and label data
    images.append(testDataLoader.dataset[i][0])
    labels.append(testDataLoader.dataset[i][1])

plt.figure(figsize=(20, 20))
columns = 5

# graph images and results
for i, image in enumerate(images):
    plt.subplot(len(images) // columns + 1, columns,  i + 1)

    # reverse normalize
    original = images * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    original = original.clamp(0, 1)

    plt.imshow(original.permute(1, 2, 0))
    plt.title(classes[resnet56(image.unsqeeze(0).to("cuda")).argmax().item()] + "true value: " + classes[labels[i]])
plt.show()
