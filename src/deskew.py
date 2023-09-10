

import torch
import os
from PIL import Image, ImageOps
from torchvision import transforms
import glob
from image_util import MAX_SKEW_ANGLE 

from model import DeskewCNN, prepare_image
from train import IMAGE_SIZE, MIN_ANGLE_ZERO_OFFSET, MyDataset


def deskew(model, image_paths, output_dir):
  for image_path in image_paths:
    image = Image.open(image_path)
    img_tensor = prepare_image(image, IMAGE_SIZE)
    x = [x for x in torch.utils.data.DataLoader(MyDataset([img_tensor]))][0]['data']
    print(f"img_tensor size: {x.size()} image_path: {image_path}")
    skew_angle_pred = model(x).argmax(dim=1) - MIN_ANGLE_ZERO_OFFSET
    rotated = image.rotate(skew_angle_pred, resample=Image.BICUBIC, expand=True, fillcolor=(255 // 2))
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    rotated.save(output_path)

if __name__ == '__main__':
  loaded_state_dict = torch.load('model.pth')
  num_classes = MIN_ANGLE_ZERO_OFFSET + MAX_SKEW_ANGLE + 1
  model = DeskewCNN(num_classes, IMAGE_SIZE)
  model.load_state_dict(loaded_state_dict)
  images_dir_path = f"{os.getcwd()}/dataset/testing_data/images"
  image_paths = glob.glob(os.path.join(images_dir_path, "*"))
  print(images_dir_path)

  output_dir_path = f"{os.getcwd()}/outputs"
  deskew(model, image_paths, output_dir_path)
