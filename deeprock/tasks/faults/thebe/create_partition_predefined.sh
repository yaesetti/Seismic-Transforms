#!/bin/bash

python create_dataset.py \
    --data_dir ../../../data/original/thebe/ \
    --output_dir processed_data \
    --patch_size 96 \
    --patch_stride 48 \
    --mask_percentage 0.03 \
    --partition_method predefined