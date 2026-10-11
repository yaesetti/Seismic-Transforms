#!/bin/bash

python create_dataset.py \
    --data_dir ../../../data/original/cigChannel/Assorted_Channel_Seismic_*/ \
    --label_dir ../../../data/original/cigChannel/Assorted_Channel_Label_*/ \
    --output_dir processed_data \
    --partition_method random \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --directions xl il \
    --seed 42