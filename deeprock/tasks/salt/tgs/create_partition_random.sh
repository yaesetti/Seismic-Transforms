#!/bin/bash

python create_dataset.py \
    --data_dir ../../../data/original/tgs/train/ \
    --output_dir processed_data \
    --partition_method random \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --seed 42