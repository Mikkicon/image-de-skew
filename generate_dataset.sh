#!/bin/bash

source_dataset_dir_name="funsd-dataset"

source_dataset_path="$(pwd)/$source_dataset_dir_name"
target_dataset_path="$(pwd)/dataset"

zip_path="$(pwd)/$source_dataset_dir_name.zip"
curl https://guillaumejaume.github.io/FUNSD/dataset.zip > "$zip_path"

echo "$zip_path"
unzip "$zip_path" -d "$source_dataset_path"

./venv/bin/python3 src/generate-dataset-skew.py "$source_dataset_path" "$target_dataset_path"



