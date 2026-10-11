#!/bin/bash

python create_dataset.py \
    --data_dir ../../../data/original/F3/data/ \
    --output_dir processed_data \
    --partition_method test_predefined_val_last \
    --n_val_samples 50 \
    --attributes envelope semblance