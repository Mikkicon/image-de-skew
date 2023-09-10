import torch
import torch.nn.functional as F
import glob 
import os
import torch
import numpy as np
import PIL.Image as Image
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from image_util import MAX_SKEW_ANGLE, MIN_SKEW_ANGLE, rotate
from model import DeskewCNN, prepare_image

MIN_ANGLE_ZERO_OFFSET = MIN_SKEW_ANGLE if MIN_SKEW_ANGLE >= 0 else -MIN_SKEW_ANGLE
TRAIN_SIZE = 10
IMAGE_SIZE = (500, 400)

def getXy(images_dir_path):
  images = []
  labels = []
  image_paths = glob.glob(os.path.join(images_dir_path, "*"))
  total_images = len(image_paths)
  print(f"{total_images} images from path {images_dir_path}")
  if(total_images == 0):
     raise Exception(f"No images in {images_dir_path}")
  image_paths = list(np.random.choice(image_paths, size=TRAIN_SIZE)) if TRAIN_SIZE else image_paths
  print(f"choosing {len(image_paths)} images from {total_images}")
  
  for image_path in image_paths:
    img = Image.open(image_path)
    img_tensor = prepare_image(img, IMAGE_SIZE)
    print(f"img_tensor size: {img_tensor.size()} image_path: {image_path}")
    try: 
      skew_angle_str = os.path.basename(image_path).split('_')[0]
      skew_angle = torch.tensor(float(skew_angle_str) + MIN_ANGLE_ZERO_OFFSET).to(torch.long)
    except Exception:
       continue
    images.append(img_tensor)
    labels.append(skew_angle)
  to_pil = transforms.ToPILImage()
  image = to_pil(images[0])
  image.save("output_image.png")
  return (images, labels)

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

def train(model, train_loader, criterion, optimizer):
  hypothesis = None
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
  return hypothesis

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data, target = sample['data'], sample['target']
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
  torch.set_default_device(device)
  # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
  dtype = torch.float32 if device.type == 'mps' else torch.float
  torch.set_default_dtype(dtype)
  print (f"running {device} with {dtype} dtype")

  num_classes = MIN_ANGLE_ZERO_OFFSET + MAX_SKEW_ANGLE + 1
  model = DeskewCNN(num_classes, IMAGE_SIZE)

  images_train_dir_path = f"{(os.getcwd())}/dataset/training_data/images"
  X, y = getXy(images_train_dir_path)
  images_test_dir_path = f"{(os.getcwd())}/dataset/testing_data/images"
  X_test, y_test = getXy(images_test_dir_path)
  criterion = torch.nn.CrossEntropyLoss(reduction='sum')                # TODO why cross entropy?
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)# TODO revise how it works
  train_loader = torch.utils.data.DataLoader(MyDataset(X, y))
  test_loader = torch.utils.data.DataLoader(MyDataset(X_test, y_test))
  n_epochs = 1
  scheduler = StepLR(optimizer, step_size=1) # TODO what is that
  for epoch in range(1, n_epochs + 1):
    train(model, train_loader, criterion, optimizer)
    test(model, test_loader, device)
    scheduler.step()
  torch.save(model.state_dict(), 'model.pth')
  output_dir_path = f"{(os.getcwd())}/outputs"
  # deskew(model, train_loader, output_dir_path) 

if __name__ == '__main__':
  main()
  # TODO read about autograd
  # TODO read about nograd