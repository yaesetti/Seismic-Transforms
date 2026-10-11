# Seismic Attribute Calculation (Regression)

Seismic data contains valuable information about the Earth's subsurface and plays a central role in oil and gas (O&G) exploration. Seismic attributes are derived from this data to emphasize relevant structures and properties, thereby enhancing geological and geophysical interpretation. However, when computed over large datasets, common in the O&G industry, these attributes can be computationally expensive in terms of both processing power and memory usage.
This task focuses on seismic attribute calculation formulated as a regression problem, where the objective is to predict a seismic attribute directly from seismic data. Solving this problem requires learning complex relationships between the input data and the corresponding attributes, including both local and non-local patterns present in seismic images.

For this study, we use the F3 dataset [6], which consists of a three-dimensional seismic volume divided into training and test sets. Two seismic attributes, envelope and semblance, are computed from the seismic data following the approach of Hecker et al. [7]. The files `train/train_seismic.npy` and `test_once/test1_seismic.npy` are used as the training and test inputs, respectively. The file `test_once/test2_seismic.npy` is excluded because its shape differs from that of the training data, making it unsuitable for consistent model evaluation.

The training data is further split into training and validation subsets using a predefined scheme, where the last 10% of seismic sections are reserved for validation. The test set is provided as a separate file. Seismic attributes are computed for the entire training dataset (prior to the split) and for the test dataset, producing attribute volumes with the same dimensions as the original seismic data. Finally, the volumes are sliced along the first spatial dimension to generate two-dimensional sections of size (255 \times 701) pixels. These sections are stored as individual NumPy arrays in a standardized directory structure, without additional preprocessing.

## 1) Download Original Data


Download the files from Zenodo:

https://zenodo.org/record/3755060/files/data.zip


Unzip the `data.zip` file file inside your `data/original/F3` directory. The expected structure is as follows:

```text
data/original/F3/data/
├── test_once
│   └── test1_seismic.npy
└── train
    └── train_seismic.npy
```

## 2) Create Processed Dataset

Main script: `create_dataset.py`

```bash
python create_dataset.py \
    --data_dir ../../../data/original/F3/data/ \
    --output_dir processed_data \
    --partition_method test_predefined_val_last \
    --n_val_samples 50 \
    --attributes envelope
```

Or run the convenience script:

```bash
bash create_partition_test_predefined_val_last.sh
```

What this script does:

- Loads train (`train/train_seismic.npy`) and test (`test_once/test1_seismic.npy`) seismic data from .npy files
- Reorders dimensions from Z, X, Y to X, Z, Y for storage
- Calculates seismic attributes for train and test cubes
- Split train data into training and validation sets, using the specified partition method. Test set is not modified and remains the same.
- Saves dataset statistics in JSON

Output structure:

```text
processed_data/
├── partition_method_test_predefined_val_last_stats.json
└── partition_method_test_predefined_val_last/
    ├── train/
    │   ├── data/
    │   └── attributes/
    │       └── envelope/
    ├── val/
    │   ├── data/
    │   └── attributes/
    │       └── envelope/
    └── test/
        ├── data/
        └── attributes/
            └── envelope/
```

Notes:

- The `data` folders contain seismic 2D slices as .npy files.
- The `attributes` folders contain seismic attributes as .npy files, organized by attribute type (e.g. `attributes/envelope/000.npy`).
- We only use the `test1_seismic.npy` file for testing, while `test2_seismic.npy` is not used because its shape differs from training samples.
- The label files (`train/train_labels.npy` and `test_once/test1_labels.npy`) are not used in this task, as we are only interested in calculating seismic attributes. They can be ignored or deleted.

## 3) Plot Processed Samples

Plot script: `plot_dataset.py`

You can select exact files:

```bash
python plot_dataset.py \
    --processed_dir processed_data/partition_method_test_predefined_val_last \
    --split train \
    --filenames 000 100 300 \
    --attributes envelope \
    --output_dir figures/selected_samples
```

Or plot first N files automatically:

```bash
python plot_dataset.py \
    --processed_dir processed_data/partition_method_test_predefined_val_last \
    --split train \
    --max_samples 4 \
    --attributes envelope \
    --output_dir figures/selected_samples
```

If you generated multiple attributes in preprocessing (for example `--attributes envelope semblance`), you can plot both by passing both names:

```bash
python plot_dataset.py \
    --processed_dir processed_data/partition_method_test_predefined_val_last \
    --split train \
    --max_samples 2 \
    --attributes envelope semblance \
    --output_dir figures/selected_samples
```

Convenience script:

```bash
bash plot_some_samples.sh
```

## Supported Partition Methods

The create_dataset.py script supports partitioning strategies through `--partition_method`. For this dataset, we currently support:

- `test_predefined_val_last`: use all provided test data for test split, and reserve the last `n_val_samples` from train data for validation. Test is a separate cube provided in the original data and is not modified.
