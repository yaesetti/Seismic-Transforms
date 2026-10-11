# Seismic Super-Resolution Dataset (Regression)

Seismic interpretation is often limited by the quality of acquired data, which typically suffers from low spatial resolution and noise contamination due to acquisition and processing constraints. These limitations difficults the identification of fine geological structures such as thin layers and small-scale faults. Traditional approaches to improve resolution, such as high-density acquisition or broadband processing, are costly and computationally demanding. To address these challenges, recent work has proposed deep learning methods capable of performing seismic image super-resolution and denoising simultaneously [8]. In particular, convolutional neural networks (CNNs) can learn to reconstruct high-resolution, clean seismic images directly from low-resolution, noisy inputs, significantly enhancing structural details while preserving geological consistency .

The dataset used for this task follows the methodology proposed by Li et al. [8]. Synthetic datasets are generated to serve as ground truth. Specifically, three-dimensional seismic volumes are simulated by constructing realistic reflectivity models that include geological features such as folding and faulting. These models are convolved with high-frequency wavelets to produce high-resolution seismic volumes, from which two-dimensional slices are extracted as target images. Corresponding low-resolution inputs are created by applying low-frequency wavelets, adding random noise, and downsampling the resulting images. This process yields paired datasets of noisy low-resolution inputs and clean high-resolution targets, enabling supervised learning of the super-resolution mapping.

The dataset is composed of multiple independent 2D seismic images, allowing for a random partitioning strategy. Each sample consists of a low-resolution input of size ($128 \times 128$) pixels and a corresponding high-resolution target of size ($256 \times 256$), reflecting an upsampling factor of 2. The data is randomly split into training, validation, and test sets. This random partitioning is appropriate because the samples are extracted from different regions and volumes, reducing spatial correlation between subsets. After that, images are stored as individual NumPy arrays in a standardized directory structure, without additional preprocessing.


## 1) Download Original Data


Download the files from Zenodo:

https://drive.google.com/drive/folders/1DuMdclOdeXDgGBOhsHSlEdTB_LvhIH-X?usp=sharing


Unzip th file file inside your `data/original/seismicSuperResolutionData` directory. The expected structure is as follows:

```text
data/original/seismicSuperResolutionData/
├── nx2
│   ├── 0000.dat
│   └── ... (more .dat files)
└── sx
    ├── 0000.dat
    └── ... (more .dat files)
```

## 2) Create Processed Dataset

Main script: `create_dataset.py`

```bash
python create_dataset.py \
    --data_dir ../../../data/seismicSuperResolutionData \
    --output_dir processed_data \
    --partition_method random \
    --percent_val 10 \
    --percent_test 20
```

Or run the convenience script:

```bash
bash create_partition_random.sh
```

What this script does:

- Loads the original .dat files from nx2 (low-res) and sx (high-res) directories.
- Reshapes the data into 2D seismic slices (128x128 for low-res, 256x256 for high-res).
- Partitions the data into train/val/test splits according to the specified method (e.g. random partitioning).
- Calculate some statistics about the splits (e.g. mean/std) and saves them in a JSON file.
- Save the processed seismic slices as .npy files in a structured directory format.

Output structure:

```text
processed_data/
├── partition_method_random.json
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

Notes:

- The `data` folders contain seismic 2D slices as .npy files (low-res) while the `label` folders contain the corresponding high-res slices as .npy files, organized by sample name (e.g. `data/0000.npy` and `label/0000.npy`).

## 3) Plot Processed Samples

Plot script: `plot_dataset.py`

You can select exact files:

```bash
python plot_dataset.py \
    --processed_dir processed_data/partition_method_random \
    --split train \
    --filenames 0000 1000 0300 \
    --output_dir figures/selected_samples
```

Or plot first N files automatically:

```bash
python plot_dataset.py \
	--processed_dir processed_data/partition_method_random \
	--split train \
	--max_samples 10 \
	--output_dir figures/selected_samples
```

Convenience script:

```bash
bash plot_some_samples.sh
```

## Supported Partition Methods

The create_dataset.py script supports partitioning strategies through `--partition_method`. For this dataset, we currently support:

- `random`:create random train/val/test splits based on specified percentages (e.g. 70% train, 10% val, 20% test).
