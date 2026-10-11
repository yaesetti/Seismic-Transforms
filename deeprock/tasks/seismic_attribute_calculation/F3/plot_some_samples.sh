#!/bin/bash

python plot_dataset.py \
	--processed_dir processed_data/partition_method_test_predefined_val_last \
	--split train \
	--max_samples 10 \
	--attributes envelope semblance \
	--output_dir figures/selected_samples

python plot_dataset.py \
	--processed_dir processed_data/partition_method_test_predefined_val_last \
	--split val \
	--max_samples 4 \
	--attributes envelope semblance \
	--output_dir figures/selected_samples

python plot_dataset.py \
	--processed_dir processed_data/partition_method_test_predefined_val_last \
	--split test \
	--max_samples 4 \
	--attributes envelope semblance \
	--output_dir figures/selected_samples

