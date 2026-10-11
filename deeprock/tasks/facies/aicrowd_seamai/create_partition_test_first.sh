#!/bin/bash

python create_dataset.py \
    --data_dir ../../../data/original/aicrowd_seamai/ \
    --output_dir processed_data \
    --partition_method test_first \
    --n_val_samples 51 \
    --n_test_samples 200