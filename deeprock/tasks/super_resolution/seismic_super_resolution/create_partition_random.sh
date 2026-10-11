#!/bin/bash

python create_dataset.py \
    --data_dir ../../../data/original/seismicSuperResolutionData/ \
    --output_dir processed_data \
    --partition_method random \
    --percent_val 10 \
    --percent_test 20 \
    --seed 42