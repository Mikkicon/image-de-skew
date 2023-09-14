import cv2
import numpy as np
import math
import torch
import os
import glob 
from typing import Tuple, Union, List
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms


load_dotenv()

MIN_SKEW_ANGLE = -30
MAX_SKEW_ANGLE = 30
FILENAME_ANGLE_SPLITTER = "_"
ANGLE_AT_START = False

MIN_ANGLE_ZERO_OFFSET = MIN_SKEW_ANGLE if MIN_SKEW_ANGLE >= 0 else -MIN_SKEW_ANGLE
N_NN_OUTPUT_CLASSES = MIN_ANGLE_ZERO_OFFSET + MAX_SKEW_ANGLE + 1
TARGET_ZEROS = [0 for idx in range(0, N_NN_OUTPUT_CLASSES)]

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu') # As I was implementing this on MAC - I used mps, but in container only able to use cpu
# TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
dtype = torch.float32 if DEVICE.type == 'mps' else torch.float
torch.set_default_dtype(dtype)
print (f"DEVICE - {DEVICE}; DTYPE - {dtype}")

BATCHSIZE = 16
N_TRAIN_EXAMPLES = BATCHSIZE * 20
N_VALID_EXAMPLES = BATCHSIZE * 10
IMAGE_SIZE = (500, 400)

SKEW_FILL_COLOR = (255, 255, 255)

def get_skew_angle_from_path(image_path) -> float:
    try: 
      skew_angle_str = os.path.basename(image_path).split('_')[0] if ANGLE_AT_START else os.path.basename(image_path).split('_')[-1].split('.')[0]
      skew_angle = torch.tensor(float(skew_angle_str)).to(torch.long)
      if(skew_angle < MIN_SKEW_ANGLE or skew_angle > MAX_SKEW_ANGLE):
         print(f"Invalid skew angle: {skew_angle}")
         return None
      return skew_angle
    except Exception:
       return None

def get_path_with_skew_angle(file_path, angle):
    filename, extension = os.path.splitext(os.path.basename(file_path))
    if ANGLE_AT_START:
        return f"{angle}{FILENAME_ANGLE_SPLITTER}{filename}{extension}"
    else:
        return f"{filename}{FILENAME_ANGLE_SPLITTER}{angle}{extension}"

def save_plot(ys, xs = None):
    plt.figure(figsize=(len(ys), 16))
    if not xs:
        xs = range(len(ys))
    plt.plot(xs,ys,  label='My Data', marker='o', linestyle='-', color='b')
    with open('output/losses.txt', 'w') as file:
        file.writelines([str(x) for x in ys])
    plt.savefig('output/line_chart.png', dpi=300) 

def save_image_grid(imgs, rows, cols, name):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    grid.save(f"output/layers/{name}")

def prepare_image(image, image_size):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),                     
        transforms.Resize(image_size, antialias=True),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img = image.convert('L')                                              
    return transform(img)

def image_to_tensor( image_path):
    if not os.path.isfile(image_path):
       return 
    img = Image.open(image_path)
    img_tensor = prepare_image(img, IMAGE_SIZE)
    skew_angle = get_skew_angle_from_path(image_path)
    if skew_angle == None:
       return
    return (img_tensor, skew_angle)

def getXy(images_dir_path):
  images = []
  labels = []
  image_paths = glob.glob(os.path.join(images_dir_path, "*"))
  total_images = len(image_paths)
  if(total_images == 0):
     raise Exception(f"No images in {images_dir_path}")
  image_paths = list(np.random.choice(image_paths, size=N_TRAIN_EXAMPLES)) if N_TRAIN_EXAMPLES else image_paths
  print(f"choosing {len(image_paths)} images from {total_images}")
  for image_path in image_paths:
     result = image_to_tensor(image_path)
     if not result:
        continue
     images.append(result[0])
     labels.append(result[1])
  print(f"Processed {len(images)} images and {len(labels)} labels")
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
        sample["target"] = self.labels[idx] + MIN_ANGLE_ZERO_OFFSET if idx < len(self.labels) else []
        return sample

def get_train_test_dataset():
  X, y = getXy(TRAIN_DIR_PATH)
  X_test, y_test = getXy(TEST_DIR_PATH)
  if not(len(X)) or not(len(y)) or not(len(X_test)) or not(len(y_test)):
     raise Exception(f"No train or test data for paths: {TRAIN_DIR_PATH} {TEST_DIR_PATH}")
  train_loader = DataLoader(MyDataset(X, y), batch_size=BATCHSIZE)
  test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=BATCHSIZE)
  return (train_loader, test_loader)

CWD = os.getcwd()

RAW_DATASET_PATH = os.path.join(CWD, os.getenv('RAW_DATASET_PATH'))
RAW_TRAIN_DIR_PATH = os.path.join(RAW_DATASET_PATH,  os.path.join('training_data', 'images'))
RAW_TEST_DIR_PATH = os.path.join(RAW_DATASET_PATH,  os.path.join('testing_data', 'images') )

DATASET_DIR_PATH = os.path.join(CWD, os.getenv('DATASET_DIR_NAME'))

TRAIN_DIR_PATH = os.path.join(DATASET_DIR_PATH, os.path.join('training_data', 'images'))
TEST_DIR_PATH = os.path.join(DATASET_DIR_PATH, os.path.join('testing_data', 'images') )

INVOICES_DIR_PATH = os.path.join(CWD, 'invoices_rotated', 'images')
OUTPUT_DIR_PATH = os.path.join(CWD, os.getenv('OUTPUT_DIR_NAME', 'output' ))

EPOCHS = int(os.getenv('EPOCHS',  '1'))

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)
