import cv2
import os
import time
from typing import List
import glob
from multiprocessing import Pool, cpu_count
import random

from image_util import rotate

MIN_SKEW_ANGLE = -30
MAX_SKEW_ANGLE = 30

def skew(file_path, output_dir):
  image = cv2.imread(file_path)
  angle = get_random_skew_angle(MIN_SKEW_ANGLE, MAX_SKEW_ANGLE)
  rotated = rotate(image, angle, (0, 0, 0))
  output_path = os.path.join(output_dir, f"{angle}_{os.path.basename(file_path)}")
  cv2.imwrite(output_path, rotated)

def get_random_skew_angle(min: int, max: int):
   return round(random.uniform(min, max),3 )

def process_files(files: List[str], output_dir: str):
  if not os.path.isdir(output_dir):
     os.mkdir(output_dir)
  for idx, file_path in enumerate(files):
    print(f"Processing {file_path} {idx + 1}/{len(files)}")
    skew(file_path, output_dir)

def main():
  start = time.time()

  training_files = glob.glob(os.path.join(f"{os.getcwd()}/funsd-dataset/training_data/deskewed_images", "*"))
  testing_files = glob.glob(os.path.join(f"{os.getcwd()}/funsd-dataset/testing_data/deskewed_images", "*"))
  training_output_dir = f"{os.getcwd()}/dataset/training_data/images"
  testing_output_dir = f"{os.getcwd()}/dataset/testing_data/images"

  if len(training_files) == 0 or len(testing_files) == 0:
    raise Exception(f"No train ({len(training_files)}) or test ({len(testing_files)}) data")

  with Pool(cpu_count()) as pool:
      pool.starmap(process_files, [(training_files, training_output_dir), (testing_files, testing_output_dir)])
  print(f"Execution took {(time.time() - start)}s")

if __name__ == '__main__':
  main()