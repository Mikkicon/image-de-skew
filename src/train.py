
import torch
import torch.nn.functional as F
import glob 
import os
import torch
import numpy as np
import PIL.Image as Image
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from multiprocessing import Pool, Manager, cpu_count
import functools
import torch.optim as optim

from image_util import IMAGE_SIZE, N_EPOCHS, N_NN_OUTPUT_CLASSES, TEST_DIR_PATH, TRAIN_DIR_PATH, TRAIN_SIZE, get_skew_angle_from_path
from model import DeskewCNN, MyDataset, prepare_image

def image_to_tensor(images, labels, image_path):
    img = Image.open(image_path)
    img_tensor = prepare_image(img, IMAGE_SIZE)
    skew_angle = get_skew_angle_from_path(image_path)
    if skew_angle == None:
       return
    images.append(img_tensor)
    labels.append(skew_angle)

def getXy(images_dir_path):
  manager = Manager()
  images = manager.list()
  labels = manager.list()
  image_paths = glob.glob(os.path.join(images_dir_path, "*"))
  total_images = len(image_paths)
  if(total_images == 0):
     raise Exception(f"No images in {images_dir_path}")
  
  image_paths = list(np.random.choice(image_paths, size=TRAIN_SIZE)) if TRAIN_SIZE else image_paths
  print(f"choosing {len(image_paths)} images from {total_images}")
  
  with Pool(cpu_count()) as pool:
    pool.starmap(functools.partial(image_to_tensor, images, labels),[[p] for p in image_paths])

  print(f"Processed {len(images)} images and {len(labels)} labels")
  return (images, labels)

def train(model: torch.nn.Module, train_loader, optimizer, epoch):
  model.train()
  for idx, sample in enumerate(train_loader):
      x, y = sample['data'], sample['target']
      optimizer.zero_grad()                                             
      hypothesis = model(x)
      loss = F.nll_loss(hypothesis, y)
      loss.backward()
      if idx % 40 == 0:
        print(f"epoch {epoch} loss {loss} hypothesis: {hypothesis.argmax(dim=1).item()} y: {y.item()}")
      optimizer.step()

def test(model: torch.nn.Module, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data, target = sample['data'].to(device), sample['target'].to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= max(len(test_loader.dataset), 1)
    print(f"nTest set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)")

def main():
  # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
  device = torch.device('cpu') # As I was implementing this on MAC - I used mps, but in container only able to use cpu
  torch.set_default_device(device)
  # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
  dtype = torch.float32 if device.type == 'mps' else torch.float
  torch.set_default_dtype(dtype)
  print (f"DEVICE - {device}; DTYPE - {dtype}")
  
  torch.manual_seed(42)

  model = DeskewCNN(N_NN_OUTPUT_CLASSES, IMAGE_SIZE)

  X, y = getXy(TRAIN_DIR_PATH)
  X_test, y_test = getXy(TEST_DIR_PATH)

  if not(len(X)) or not(len(y)) or not(len(X_test)) or not(len(y_test)):
     raise Exception(f"No train or test data for paths: {TRAIN_DIR_PATH} {TEST_DIR_PATH}")

  # criterion = torch.nn.CrossEntropyLoss(reduction='sum')                
  optimizer = optim.Adadelta(model.parameters(), lr=1.0)
  train_loader = torch.utils.data.DataLoader(MyDataset(X, y))
  test_loader = torch.utils.data.DataLoader(MyDataset(X_test, y_test))
  
  scheduler = StepLR(optimizer, step_size=1)
  for epoch in range(1, N_EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader, device)
    scheduler.step()
  torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
  main()
