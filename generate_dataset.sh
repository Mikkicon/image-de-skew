#!/bin/bash


RAW_DATASET_DIR_NAME="funsd-dataset"
zip_path="dataset"

rm -r "$zip_path"
rm -r "$RAW_DATASET_DIR_NAME"

curl https://guillaumejaume.github.io/FUNSD/dataset.zip > "$zip_path.zip"

unzip -q "$zip_path"  -d "$RAW_DATASET_DIR_NAME"

if [ -d "./$RAW_DATASET_DIR_NAME/__MACOSX" ]; then
  rm -r "./$RAW_DATASET_DIR_NAME/__MACOSX"
  mv ./$RAW_DATASET_DIR_NAME/dataset/* ./$RAW_DATASET_DIR_NAME
  rm -r "./$RAW_DATASET_DIR_NAME/dataset"
fi

python3 src/generate-dataset-skew.py
