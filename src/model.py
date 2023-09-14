import torch
import torch.nn as nn
import torch
from torchvision import transforms

from image_util import MIN_ANGLE_ZERO_OFFSET, N_NN_OUTPUT_CLASSES, TARGET_ZEROS, save_image_grid

to_pil = transforms.ToPILImage()

class DeskewCNN(nn.Module):
    def __init__(self, num_classes, image_size):
        super(DeskewCNN, self).__init__()
        
        # hyperparameters
        conv_kernel = 3
        pool_kernel_1 = 2
        pool_kernel_2 = 2

        # first convolution of layer expects image in grayscale and outputs 12 feature kernels
        self.conv1 = nn.Conv2d(1, 12, kernel_size=conv_kernel, stride=1, padding=(conv_kernel - 1) // 2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_1, stride=pool_kernel_1)

        # in first convolution layer we are able to distinguish separate letters
        # so we do a max pooling which comprises the first convolution layer output
        # second convolution is now able to distinguish the line angle in which the words are written
        self.conv2 = nn.Conv2d(12, 24, kernel_size=conv_kernel, stride=1, padding=(conv_kernel - 1) // 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_2, stride=pool_kernel_2)

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(24 * (image_size[0] // (pool_kernel_1 * pool_kernel_2)) * (image_size[1] // (pool_kernel_1 * pool_kernel_2)), 128, bias=True)

        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes, bias=True)

        print(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        save_image_grid([to_pil(t) for t in x.squeeze()], 3, 4, f"{'train_' if self.training else 'test_' }pool1.png")

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        save_image_grid([to_pil(t) for t in x.squeeze()], 4, 6, f"{'train_' if self.training else 'test_' }pool2.png")

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        return x

def prepare_image(image, image_size):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),                     
        transforms.Resize(image_size, antialias=True),
        transforms.Normalize((0.1307,), (0.3081,))
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
        sample["target"] = self.labels[idx] + MIN_ANGLE_ZERO_OFFSET if idx < len(self.labels) else []
        return sample
