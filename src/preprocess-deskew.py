import cv2
import os
import time
from typing import List
import glob
from multiprocessing import Pool, cpu_count
from deskew import determine_skew

from image_util import rotate


def deskew(file_path, output_dir):
  image = cv2.imread(file_path)
  grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  angle = determine_skew(grayscale)
  rotated = rotate(image, angle, (0, 0, 0))
  output_path = os.path.join(output_dir, os.path.basename(file_path))
  cv2.imwrite(output_path, rotated)

def process_files(files: List[str], output_dir: str):
  if not os.path.isdir(output_dir):
     os.mkdir(output_dir)
     
  for idx, file_path in enumerate(files):
    print(f"Processing {file_path} {idx + 1}/{len(files)}")
    deskew(file_path, output_dir)

def main():
  start = time.time()

  training_files = glob.glob(os.path.join(f"{os.getcwd()}/funsd-dataset/training_data/images", "*"))
  testing_files = glob.glob(os.path.join(f"{os.getcwd()}/funsd-dataset/testing_data/images", "*"))
  training_output_dir = f"{os.getcwd()}/funsd-dataset/training_data/deskewed_images"
  testing_output_dir = f"{os.getcwd()}/funsd-dataset/testing_data/deskewed_images"

  if len(training_files) == 0 or len(testing_files) == 0:
    raise Exception(f"No train ({len(training_files)}) or test ({len(testing_files)}) data")

  # Execution took 15.209922075271606s
  # process_files(testing_files, testing_output_dir)
  # files_per_chunk = 20
  # args = training_files[::files_per_chunk]
  with Pool(cpu_count()) as pool:
      result = pool.starmap(process_files, [(training_files, training_output_dir), (testing_files, testing_output_dir)])

  print(f"Execution took {(time.time() - start)}s")


if __name__ == '__main__':
  main()