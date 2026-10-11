#!/bin/bash

python plot_dataset.py \
    --processed_dir processed_data/partition_method_predefined \
    --split train \
    --max_samples 4 \
    --output_dir figures/selected_samples

python plot_dataset.py \
    --processed_dir processed_data/partition_method_predefined \
    --split val \
    --max_samples 4 \
    --output_dir figures/selected_samples

python plot_dataset.py \
    --processed_dir processed_data/partition_method_predefined \
    --split test \
    --max_samples 4 \
    --output_dir figures/selected_samples
