# Channel Identification (Binary Segmentation)

This task focuses on binary semantic segmentation of seismic data, where each pixel is assigned to either a channel or non-channel class. Channel identification requires capturing both local reflector textures and broader depositional geometry, because channels can be laterally continuous, branching, and morphologically complex across the seismic cube.

For this task, we use the CIG Channel dataset, which provides 100 3D seismic cubes and corresponding channel labels in `.dat` format. Each cube has dimensions of `(256, 256, 256)` and is labeled with multiple channel classes. However, for our benchmark, we remap all channel labels to a single binary class (channel vs non-channel) to focus on the core segmentation challenge. This dataset was sinthetically generated using a geological process-based model and is designed to mimic real-world channel features while providing clean ground truth labels.

We perform a random train/val/test split at the cube level, resulting in 70 cubes for training, 10 for validation, and 20 for testing. Each cube is then sliced into 2D images along the depth axis, yielding a large number of 2D samples for model training and evaluation. No additional preprocessing is applied, and both seismic data and labels are stored in NumPy format for efficient loading.

## 1) Download Original Data

Download the CIG Channel dataset from:

https://zenodo.org/records/11044512

Or using the following bash script:

```bash
for i in "0-49" "50-99" "100-149" "150-199" "200-249" "250-299" "300-349" "350-399"; do
    for j in "Seismic" "Label"; do
        echo "-------- Downloading Assorted_Channel_${j}_${i}..."
        wget -O Assorted_Channel_${j}_${i}.zip https://zenodo.org/records/11044512/files/Assorted_Channel_${j}_${i}.zip?download=1
        mkdir -p Assorted_Channel_${j}_${i}
        unzip -o Assorted_Channel_${j}_${i}.zip -d Assorted_Channel_${j}_${i}
        rm Assorted_Channel_${j}_${i}.zip
        echo "-------- Finished downloading Assorted_Channel_${j}_${i}."
    done
done
```

Place (or symlink) the original `.dat` files under `data/original/cigChannel` with the following structure:

```text
data/original/cigChannel/
├── Assorted_Channel_Seismic_0-49/
│   └── *.dat
├── Assorted_Channel_Seismic_50-99/
│   └── *.dat
├── Assorted_Channel_Label_0-49/
│   └── *.dat
└── Assorted_Channel_Label_50-99/
	└── *.dat
	...
```

Notes:

- Seismic and label files are paired by filename stem.
- Each `.dat` file is loaded with shape `(256, 256, 256)`.

## 2) Create Processed Dataset

Main script: `create_dataset.py`

```bash
python create_dataset.py \
	--data_dir ../../../data/original/cigChannel/Assorted_Channel_Seismic_300-349/ ../../../data/original/cigChannel/Assorted_Channel_Seismic_350-399/ \
	--label_dir ../../../data/original/cigChannel/Assorted_Channel_Label_300-349/ ../../../data/original/cigChannel/Assorted_Channel_Label_350-399/ \
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

- Loads seismic cubes (`float32`) and label cubes (`uint16`) from `.dat` files.
- Remaps labels to binary format (0 for non-channel, 1 for channel) using a simple thresholding function.
- Reorders dimensions from `Z, X, Y` to `X, Z, Y` for storage.
- Splits samples into train/val/test at the cube level.
- Stacks and flattens split cubes into 2D slices of shape `(256, 256)`.
- Saves data/label pairs as `.npy` files.
- Saves dataset statistics in JSON.

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
	--filenames 00000 00120 01024 \
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

- `random`: random train/val/test split using sklearn `train_test_split`.
