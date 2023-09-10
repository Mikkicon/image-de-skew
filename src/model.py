import torch
import torch.nn as nn
import glob 
import os
import torch
import numpy as np
import PIL.Image as Image
from torchvision import transforms

class DeskewCNN(nn.Module):
    def __init__(self, num_classes, image_size):
        super(DeskewCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(24 * (image_size[0] // 4) * (image_size[1] // 4), 128) # TODO find correlation
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # print("conv1 input size:", x.size())
        x = self.conv1(x)
        # print("relu1 input size:", x.size())
        x = self.relu1(x)
        # print("pool1 input size:", x.size())
        x = self.pool1(x)
        # print("conv2 input size:", x.size())
        x = self.conv2(x)
        # print("relu2 input size:", x.size())
        x = self.relu2(x)
        # print("pool2 input size:", x.size())
        x = self.pool2(x)
        # print("flatten input size:", x.size())
        x = self.flatten(x)
        # print("fc1 input size:", x.size())
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def prepare_image(image, image_size):
    transform = transforms.Compose([
        transforms.PILToTensor(),                                         # TODO How Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
        transforms.ConvertImageDtype(torch.float32),                      # TODO __REMOVE__ - MAC stuff
        transforms.Resize(image_size)                                     # resize
    ])
    img = image.convert('L')                                              # grayscale
    return transform(img)