import sys
import cv2
import os
import time
from typing import List
import glob
from multiprocessing import Pool, cpu_count
import random

from image_util import  MAX_SKEW_ANGLE, MIN_SKEW_ANGLE, RAW_TEST_DIR_PATH, RAW_TRAIN_DIR_PATH, TEST_DIR_PATH, TRAIN_DIR_PATH, get_path_with_skew_angle, rotate

def skew(file_path, output_dir):
  image = cv2.imread(file_path)
  angle = get_random_skew_angle(MIN_SKEW_ANGLE, MAX_SKEW_ANGLE)
  rotated = rotate(image, angle, (255, 255, 255))
  output_name = get_path_with_skew_angle(file_path, angle)
  output_path = os.path.join(output_dir, output_name)
  cv2.imwrite(output_path, rotated)

def get_random_skew_angle(min: int, max: int):
   return round(random.uniform(min, max),3 )

def process_files(files: List[str], output_dir: str):
  os.makedirs(output_dir, exist_ok=True)
  for idx, file_path in enumerate(files):
    print(f"Processing {file_path} {idx + 1}/{len(files)}")
    skew(file_path, output_dir)

def main():
  start = time.time()
  training_files = glob.glob(os.path.join(RAW_TRAIN_DIR_PATH, "*"))
  testing_files = glob.glob(os.path.join(RAW_TEST_DIR_PATH, "*"))

  if len(training_files) == 0 or len(testing_files) == 0:
    raise Exception(f"No train ({len(training_files)}) or test ({len(testing_files)}) data in {RAW_TRAIN_DIR_PATH} {RAW_TEST_DIR_PATH}")

  with Pool(cpu_count()) as pool:
      pool.starmap(process_files, [(training_files, TRAIN_DIR_PATH), (testing_files, TEST_DIR_PATH)])
  print(f"Execution took {(time.time() - start)}s")

if __name__ == '__main__':
  main()