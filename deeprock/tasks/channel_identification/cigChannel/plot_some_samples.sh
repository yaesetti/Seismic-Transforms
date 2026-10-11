#!/bin/bash

python plot_dataset.py \
    --processed_dir processed_data/partition_method_random \
    --split train \
    --max_samples 10 \
    --random \
    --output_dir figures/selected_samples

python plot_dataset.py \
    --processed_dir processed_data/partition_method_random \
    --split val \
    --max_samples 4 \
    --random \
    --output_dir figures/selected_samples

python plot_dataset.py \
    --processed_dir processed_data/partition_method_random \
    --split test \
    --max_samples 4 \
    --random \
    --output_dir figures/selected_samples
