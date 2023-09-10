#!/bin/bash


RAW_DATASET_DIR_NAME="funsd-dataset"

rm -r "dataset"
rm -r "$RAW_DATASET_DIR_NAME"

curl https://guillaumejaume.github.io/FUNSD/dataset.zip

unzip -q "$zip_path"  -d "$RAW_DATASET_DIR_NAME"

if [ -d "./$RAW_DATASET_DIR_NAME/__MACOSX" ]; then
  rm -r "./$RAW_DATASET_DIR_NAME/__MACOSX"
  mv ./$RAW_DATASET_DIR_NAME/dataset/* ./$RAW_DATASET_DIR_NAME
  rm -r "./$RAW_DATASET_DIR_NAME/dataset"
fi

./venv/bin/python3 src/generate-dataset-skew.py



