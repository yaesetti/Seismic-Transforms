#!/bin/bash

python plot_dataset.py \
    --processed_dir processed_data/partition_method_test_first \
    --split train \
    --filenames il_000 il_100 il_200 il_300 il_400 il_500 il_589 xl_000 xl_100 xl_200  xl_300 xl_400 xl_500 xl_530 \
    --output_dir figures/selected_samples

python plot_dataset.py \
    --processed_dir processed_data/partition_method_test_first \
    --split val \
    --filenames xl_000 xl_051 \
    --output_dir figures/selected_samples

python plot_dataset.py \
    --processed_dir processed_data/partition_method_test_first \
    --split test \
    --filenames xl_000 xl_100 xl_199 \
    --output_dir figures/selected_samples