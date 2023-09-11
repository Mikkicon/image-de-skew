import torch
import torch.nn as nn
import torch
from torchvision import transforms

class DeskewCNN(nn.Module):
    def __init__(self, num_classes, image_size):
        super(DeskewCNN, self).__init__()
        
        # first convolution of layer expects image in grayscale and outputs 12 feature kernels
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # in first convolution layer we are able to distinguish separate letters
        # so we do a max pooling which comprises the first convolution layer output
        # second convolution is now able to distinguish the line angle in which the words are written
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(24 * (image_size[0] // 4) * (image_size[1] // 4), 128)

        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

        print(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        return x

def prepare_image(image, image_size):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),                     
        transforms.Resize(image_size)
    ])
    img = image.convert('L')                                              
    return transform(img)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels = []):
        super(MyDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = {"data": self.images[idx]}
        sample["target"] = self.labels[idx] if idx < len(self.labels) else []
        return sample
