

import torch
import os
from PIL import Image, ImageOps
import glob
import shutil
from torchvision import transforms

from image_util import  IMAGE_SIZE, INVOICES_DIR_PATH, MIN_ANGLE_ZERO_OFFSET, N_NN_OUTPUT_CLASSES, OUTPUT_DIR_PATH, SKEW_FILL_COLOR, TEST_DIR_PATH, MyDataset, prepare_image
from model import DeskewCNN 


def deskew(model, image_paths, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  for image_path in image_paths:
    image = Image.open(image_path)
    img_tensor = prepare_image(image, IMAGE_SIZE)
    x = [x for x in torch.utils.data.DataLoader(MyDataset([img_tensor]))][0]['data']
    print(f"argmax: {model(x).argmax(dim=1)} image name: {os.path.basename(image_path)}")
    skew_angle_pred = model(x).argmax(dim=1) - MIN_ANGLE_ZERO_OFFSET
    rotated = image.rotate(skew_angle_pred, resample=Image.BICUBIC, expand=True, fillcolor=SKEW_FILL_COLOR)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    rotated.save(output_path)

if __name__ == '__main__':
  loaded_state_dict = torch.load('model.pth')
  model = DeskewCNN(N_NN_OUTPUT_CLASSES, IMAGE_SIZE)
  model.load_state_dict(loaded_state_dict)

  shutil.rmtree(OUTPUT_DIR_PATH, True)

  test_image_paths = glob.glob(os.path.join(TEST_DIR_PATH, "*"))
  deskew(model, test_image_paths, OUTPUT_DIR_PATH)

  invoices_image_paths = glob.glob(os.path.join(INVOICES_DIR_PATH, "*"))
  deskew(model, invoices_image_paths, OUTPUT_DIR_PATH)
