# Facies Segmentation (Multi-class Segmentation)

This task focuses on multi-class semantic segmentation of seismic images, where each pixel is assigned a seismic facies label. Seismic facies segmentation requires modeling both fine-grained textural characteristics and large-scale stratigraphic context, as facies boundaries are often laterally continuous and extend over long spatial ranges within the seismic volume.

For this task, we use the AIcrowd SEAM-AI dataset, which comprises a three-dimensional seismic volume with corresponding facies annotations. Although the dataset provides one training volume and two test volumes, only the training volume contains labels and is therefore used in this work. To enable supervised learning and evaluation for downstream tasks, the labeled volume is deterministically partitioned along the crossline direction into three contiguous subsets, following the same partitioning strategy adopted by Tolstaya and Egorov.
The first block of crosslines (200 crosslines) is assigned to the test set, followed by a validation block of 51 crosslines, while the remaining crosslines are used for training.

Two-dimensional slices are extracted from the three-dimensional volume and stored as NumPy arrays. The training set includes both inline and crossline slices, whereas the validation and test sets contain only crossline slices in order to reduce spatial leakage between splits. No additional filtering, normalization, or data augmentation is applied during preprocessing.

This folder contains scripts to convert the original AICrowd SEAM-AI dataset into the processed format used by this repository.

## 1) Download Original Data


Download the training files from the official challenge page:

https://www.aicrowd.com/challenges/seismic-facies-identification-challenge

Place (or symlink) these files inside your `data/original/aicrowd_seamai` directory. The expected structure is as follows:

```text
data/original/aicrowd_seamai/
├── data_train.npy
└── labels_train.npy
```

## 2) Create Processed Dataset

Main script: `create_dataset.py`

```bash
python create_dataset.py \
    --data_dir ../../../data/original/aicrowd_seamai \
    --output_dir processed_data \
    --partition_method test_first \
    --n_val_samples 51 \
    --n_test_samples 200
```

Or run the convenience script:

```bash
bash create_partition_test_first.sh
```

What this script does:

- Loads data_train.npy and labels_train.npy
- Remaps labels from 1..6 to 0..5
- Reorders dimensions from Z, X, Y to X, Z, Y for storage
- Creates train/val/test split using test-first strategy
- Saves samples as .npy in data/ and label/ folders
- Saves dataset statistics in JSON

Output structure:

```text
processed_data/
├── partition_method_test_first_stats.json
└── partition_method_test_first/
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

Notes:

- Training split includes both xl_* and il_* slices.
- Validation and test splits include xl_* slices.

## 3) Plot Processed Samples

Plot script: `plot_dataset.py`

You can select exact files:

```bash
python plot_dataset.py \
    --processed_dir processed_data/partition_method_test_first \
    --split train \
    --filenames xl_000 xl_120 il_050 \
    --output_dir figures/selected_samples
```

Or plot first N files automatically:

```bash
python plot_dataset.py \
    --processed_dir processed_data/partition_method_test_first \
    --split train \
    --max_samples 4 \
    --output_dir figures/selected_samples
```

Convenience script:

```bash
bash plot_some_samples.sh
```

## Supported Partition Methods

The create_dataset.py script supports different partitioning strategies, by specifying the --partition_method argument. For this dataset, we currently support:

- `test_first`: first samples go to test, next to validation, remaining to training.
