import torch
import torch.nn as nn
import torch.nn.functional as F
import glob 
import os
import torch
import numpy as np
import PIL.Image as Image
from torchvision import transforms

from image_util import MIN_SKEW_ANGLE

MIN_ANGLE_ZERO_OFFSET = MIN_SKEW_ANGLE if MIN_SKEW_ANGLE >= 0 else -MIN_SKEW_ANGLE
TRAIN_SIZE = None

def getXy():
  images = []
  labels = []
  images_dir_path = f"{(os.getcwd())}/dataset/training_data/images"
  image_paths = glob.glob(os.path.join(images_dir_path, "*"))
  total_images = len(image_paths)
  print(f"{total_images} images from path {images_dir_path}")
  if(total_images == 0):
     raise Exception(f"No images in {images_dir_path}")
  image_paths = list(np.random.choice(image_paths, size=TRAIN_SIZE)) if TRAIN_SIZE else image_paths
  print(f"choosing {len(image_paths)} images from {total_images}")
  transform = transforms.Compose([
      transforms.PILToTensor(),                                         # TODO How Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
      transforms.ConvertImageDtype(torch.float32),                      # TODO __REMOVE__ - MAC stuff
      transforms.Resize((500, 500))                                     # resize
  ])
  for image_path in image_paths:
    img = Image.open(image_path)
    img = img.convert('L')                                              # grayscale
    img_tensor = transform(img)
    images.append(img_tensor)
    skew_angle_str = os.path.basename(image_path).split('_')[0]
    skew_angle = torch.tensor(float(skew_angle_str) + MIN_ANGLE_ZERO_OFFSET).to(torch.long)
    print(f"Processing skew_angle {skew_angle_str} -> {skew_angle}")
    labels.append(skew_angle)
  return (images, labels)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        super(MyDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = {"data": self.images[idx], "target": self.labels[idx] }
        return sample

def train(model, X, y, criterion, optimizer):
  train_loader = torch.utils.data.DataLoader(MyDataset(X, y))
  for idx, sample in enumerate(train_loader):
      x, y = sample['data'], sample['target']
      print(idx, x.size(), y.size())
      print(next(model.parameters()).device)
      hypothesis = model(x)
      print(y.item())
      print(hypothesis.size(), y.size())
      loss = criterion(hypothesis, y)
      print(f"loss {loss} hypothesis: {hypothesis.argmax(dim=1)} y: {y}")
      optimizer.zero_grad()                                             # TODO why
      loss.backward()
      optimizer.step()

def main():
  model = torch.nn.Sequential(
      torch.nn.Conv2d(
          1,  # B&W Image 0-255
          12,  # number of kernels - we need less as we only detect vertical/horizontal/diagonal lines
          3,  # kernel size
          1,  # 1 pixel at a time
          1  # padding - kernel size / 2 - to apply kernel on borders
      ),
      torch.nn.ReLU(),                                                  # TODO why ReLU
      torch.nn.MaxPool2d(kernel_size=2, stride=2),                      # TODO what it really does
      torch.nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),      # TODO why another one? try without
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Flatten(),  # Flatten the tensor
      # Adjust the input size based on your image size
      torch.nn.Linear(24 * 125 * 125, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(
          128,  # number of kernels
          30 + 1 + 30,  # [-30, 30] degrees
          # bias=True # TODO why
      ),
  )
  X, y = getXy()
  criterion = torch.nn.CrossEntropyLoss(reduction='sum')                # TODO why cross entropy?
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)# TODO revise how it works
  train(model, X, y, criterion, optimizer)

if __name__ == '__main__':
  main()
  # TODO read about autograd
  # TODO read about nograd