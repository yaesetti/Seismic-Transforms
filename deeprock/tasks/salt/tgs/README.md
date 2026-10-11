# Salt Segmentation (Binary Segmentation)

This task consists of binary semantic segmentation of seismic images, where the objective is to assign each pixel to either a salt body or the surrounding sediment. It is a dense prediction problem that requires learning spatially coherent representations capable of capturing strong impedance contrasts, sharp boundaries, and complex salt geometries, which are known to distort seismic wave propagation.

The TGS Salt Identification Challenge dataset [5] is a widely used benchmark for salt body segmentation, originally released as part of a Kaggle competition. It consists of seismic images paired with binary masks that indicate the presence of salt bodies. The original dataset includes 4000 labeled training images and a separate set of unlabeled test images.

Since our benchmark focuses on supervised learning for downstream tasks, we utilize only the labeled portion of the dataset and construct our own training, validation, and test splits.
Specifically, the dataset comprises 4000 samples of size $101 \times 101$ pixels, which are randomly partitioned into training, validation, and test sets following an 70\%/10\%/20\% split, respectively. We adopted the random split since the dataset is not volumetric and does not contain spatially contiguous samples.
No additional preprocessing is performed, and both images and masks are stored in NumPy format for efficient data loading.

This folder contains scripts to convert the TGS dataset into the processed format used by this repository.

## 1) Download Original Data

Download TGS from:

https://github.com/nmd2k/tgs-salt-identification

Only training data is required.

Place (or symlink) these folders:

```text
data/original/
└── tgs/
    ├── train_images/
    │   ├── *.png
    │   └── ...
    └── train_masks/
        ├── *.png
        └── ...
```

## 2) Create Processed Dataset

Main script: `create_dataset.py`

```bash
python create_dataset.py \
    --data_dir ../../../data/original/tgs \
    --output_dir processed_data \
    --partition_method random \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --seed 42
```

Or run the convenience script:

```bash
bash create_partition_random.sh
```

What this script does:

- Lists files from train_images and train_masks
- Creates random train/val/test partitions
- Converts PNG images/masks to NumPy arrays
- Remaps masks to binary labels (0 and 1)
- Saves data/label pairs as .npy files
- Saves dataset statistics in JSON

Output structure:

```text
processed_data/
├── partition_method_random_stats.json
└── partition_method_random/
    ├── train/
    │   ├── data/
    │   └── label/
    ├── val/
    │   ├── data/
    │   └── label/
    └── test/
        ├── data/
        └── label/
```

## 3) Plot Processed Samples

Plot script: `plot_dataset.py`

Select specific filenames:

```bash
python plot_dataset.py \
    --processed_dir processed_data/partition_method_random \
    --split train \
    --filenames 00a3af90ab 0a18b314fc \
    --output_dir figures/selected_samples
```

Or auto-select first N:

```bash
python plot_dataset.py \
    --processed_dir processed_data/partition_method_random \
    --split train \
    --max_samples 4 \
    --output_dir figures/selected_samples
```

Convenience script:

```bash
bash plot_some_samples.sh
```

## Supported Partition Methods

The following partition methods are supported, which can be selected using the `--partition_method` argument in `create_dataset.py`:

- `random`: random train/val/test split using sklearn train_test_split.
