import cv2
import numpy as np
import math
import torch
import os
from typing import Tuple, Union, List
from dotenv import load_dotenv

load_dotenv()

MIN_SKEW_ANGLE = -30
MAX_SKEW_ANGLE = 30
FILENAME_ANGLE_SPLITTER = "_"
ANGLE_AT_START = False

MIN_ANGLE_ZERO_OFFSET = MIN_SKEW_ANGLE if MIN_SKEW_ANGLE >= 0 else -MIN_SKEW_ANGLE
N_NN_OUTPUT_CLASSES = MIN_ANGLE_ZERO_OFFSET + MAX_SKEW_ANGLE + 1

TRAIN_SIZE = 10
IMAGE_SIZE = (500, 400)

def get_skew_angle_from_path(image_path) -> float:
    try: 
      skew_angle_str = os.path.basename(image_path).split('_')[0] if ANGLE_AT_START else os.path.basename(image_path).split('_')[-1].split('.')[0]
      skew_angle = torch.tensor(float(skew_angle_str) + MIN_ANGLE_ZERO_OFFSET).to(torch.long)
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

CWD = os.getcwd()

RAW_DATASET_PATH = os.path.join(CWD, os.getenv('RAW_DATASET_PATH'))
RAW_TRAIN_DIR_PATH = os.path.join(RAW_DATASET_PATH,  os.path.join('training_data', 'images'))
RAW_TEST_DIR_PATH = os.path.join(RAW_DATASET_PATH,  os.path.join('testing_data', 'images') )

DATASET_DIR_PATH = os.path.join(CWD, os.getenv('DATASET_DIR_NAME'))

TRAIN_DIR_PATH = os.path.join(DATASET_DIR_PATH, os.getenv('TRAIN_DIR_NAME',  os.path.join('training_data', 'images')))
TEST_DIR_PATH = os.path.join(DATASET_DIR_PATH, os.getenv('TEST_DIR_NAME',  os.path.join('testing_data', 'images') ))

INVOICES_DIR_PATH = os.path.join(CWD, 'invoices_rotated', 'images')
OUTPUT_DIR_PATH = os.path.join(CWD, os.getenv('OUTPUT_DIR_NAME', 'output' ))

N_EPOCHS = int(os.getenv('N_EPOCHS',  '200'))

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
