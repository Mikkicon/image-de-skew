#!/bin/bash

# dir name for raw data from github
RAW_DATASET_DIR_NAME="funsd-dataset"
zip_path="dataset"

# delete dataset dirs if already exist
rm -r "$zip_path"
rm -r "$RAW_DATASET_DIR_NAME"

# fetch dataset for trainig with scanned documents in image format
curl https://guillaumejaume.github.io/FUNSD/dataset.zip > "$zip_path.zip"

# unzip dataset archive
unzip -q "$zip_path"  -d "$RAW_DATASET_DIR_NAME"

# MACOS appendix after unzip
if [ -d "./$RAW_DATASET_DIR_NAME/__MACOSX" ]; then
  rm -r "./$RAW_DATASET_DIR_NAME/__MACOSX"
  mv ./$RAW_DATASET_DIR_NAME/dataset/* ./$RAW_DATASET_DIR_NAME
  rm -r "./$RAW_DATASET_DIR_NAME/dataset"
fi

# run the script which skews given documents
python3 src/generate-dataset-skew.py
