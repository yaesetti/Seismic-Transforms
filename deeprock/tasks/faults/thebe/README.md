# Fault Segmentation (Binary Segmentation)

This task addresses binary semantic segmentation of seismic data, with the objective of identifying fault structures within seismic images. Fault segmentation requires sensitivity to subtle reflector discontinuities and localized structural offsets, and is particularly challenging due to severe class imbalance and the often weak and discontinuous seismic signatures of faults.

For this task, we use the Thebe dataset [3], a publicly available benchmark specifically designed for seismic fault detection. The dataset consists of nine labeled seismic volumes for training, two for validation, and three for testing, following a predefined and fixed data split. Each volume is accompanied by a three-dimensional binary fault annotation.

We adopt a preprocessing pipeline similar to that of An et al. [4], in which the 3D seismic volumes are sliced along the first spatial direction to generate two-dimensional sections. From each section, patches of size $96 \times 96$ pixels are extracted using a stride of 48 pixels in both spatial dimensions. To mitigate class imbalance in training, we only keep patches containing at least 3% fault pixels, while patches dominated by background are discarded for the train and val splits.
For the test split, for every five 2D panels, we select one, and the whole panel is stored, as no patches are discarded, allowing us to perform the extraction on the fly.
No additional preprocessing is applied, and all extracted patches and panels are stored as individual NumPy arrays within a standardized directory structure.

This folder contains scripts to convert Thebe into the processed format used by this repository.

## 1) Download Original Data

Download Thebe from:

https://doi.org/10.7910/DVN/YBYGBK

Place (or symlink) files so the folder has train/val/test splits with seismic and fault volumes:

```text
data/original/
└── thebe/
    ├── train/
    │   ├── seistrain*.npz
    │   └── faulttrain*.npz
    ├── val/
    │   ├── seisval*.npz
    │   └── faultval*.npz
    └── test/
        ├── seistest*.npz
        └── faulttest*.npz
```

## 2) Create Processed Dataset

Main script: `create_dataset.py`

```bash
python create_dataset.py \
    --data_dir ../../../data/original/thebe \
    --output_dir processed_data \
    --partition_method predefined \
    --patch_size 96 \
    --patch_stride 48 \
    --mask_percentage 0.03
```

Or run the convenience script:

```bash
bash create_partition_predefined.sh
```

What this script does:

- Reads predefined train/val/test splits from the original dataset
- Pairs seis*.npz and fault*.npz files
- Reorders dimensions from Y, X, Z to Y, Z, X for storage
- Computes statistics (max, min, mean, std) for each panel, in order to enable MinMax and Z-norm normalizations by panel. 
- Pads data and labels using 'reflect' (centered in height and width) so spatial dimensions are compatible with the selected patch_size and patch_stride.
- Extracts 2D patches from each section for train/val split, while keeping the whole panel for test split (test split is sampled every 5 panels).
- Keeps train/val patches with at least mask_percentage positive label ratio
- Keeps the whole test panel
- Saves data/label pairs as .npy files
- Saves dataset statistics in JSON

Output structure:

```text
processed_data/
├── partition_method_predefined_stats.json
└── partition_method_predefined/
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
    --processed_dir processed_data/partition_method_predefined \
    --split train \
    --filenames 0155/0672_0144 0634/1056_0480 \
    --output_dir figures/selected_samples
```

Or auto-select first N:

```bash
python plot_dataset.py \
    --processed_dir processed_data/partition_method_predefined \
    --split train \
    --max_samples 4 \
    --output_dir figures/selected_samples
```

Convenience script:

```bash
bash plot_some_samples.sh
```

Note: For this dataset, we plot the whole panels for the test partition, but when loading them with the 
specialized reader, these panels will be subdivided into patches.

## Supported Partition Methods

The following partition methods are supported, which can be selected using the `--partition_method` argument in `create_dataset.py`:

- `predefined`: use dataset-provided train/val/test split.
