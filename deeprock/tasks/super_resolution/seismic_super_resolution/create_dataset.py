"""Create the super-resolution with SeismicSuperResolution Dataset.

Tutorial
--------
1. Put original files in the input folder seismicSuperResolutionData folder, which should contain:
   - sx
   - nx2
    Sx is the high-resolution image (256, 256) and nx2 is the low-resolution image (128, 128). 
2. Run this script to create train/val/test splits in .npy format. The train/data_train.npy file will be split into train, val while the test_once/test1_seismic.npy will be used as the test set. The output will be organized in a folder structure like:
    processed_data/
    ├── partition_method_random.json
    └── partition_method_test_first/
        ├── train/
        │   ├── data/
        │   └── label/
        ├── val/
        │   ├── data/
        │   └── label/
        └── test/
            ├── data/
            └─ label/
3. Use plot_dataset.py to visualize generated samples.

Example
-------
python create_dataset.py \
    --data_dir ../../../data/seismicSuperResolutionData \
    --output_dir processed_data \
    --partition_method random \
    --percent_val 10 \
    --percent_test 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def random_split(
    num_samples: int,
    percent_val: float = 10.0,
    percent_test: float = 20.0,
    seed: int = 42,
):
    """Split data into train, val and test sets using random splitting."""
    indices = np.arange(num_samples)
    generator = np.random.default_rng(seed)
    generator.shuffle(indices)

    num_val = int(num_samples * percent_val / 100)
    num_test = int(num_samples * percent_test / 100)

    val_indices = indices[:num_val]
    test_indices = indices[num_val : num_val + num_test]
    train_indices = indices[num_val + num_test :]

    print(f"Random split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test samples.")

    return (train_indices, val_indices, test_indices)



def _default_stats(data: np.ndarray) -> dict[str, float | str | list[int]]:
    """Compute basic descriptive statistics for one array.

    Parameters
    ----------
    data : np.ndarray
        Input array used to summarize distribution and shape.

    Returns
    -------
    dict[str, float | str | list[int]]
        Dictionary containing min, max, mean, std, median, dtype, and shape.
    """
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "dtype": str(data.dtype),
        "shape": list(data.shape),
    }


def get_statistics(data_low_res: np.ndarray, data_high_res: np.ndarray):
    """Compute split statistics for seismic samples and corresponding labels.
    Parameters
    ----------
    data_low_res : np.ndarray
        Low-resolution seismic data.
    data_high_res : np.ndarray
        High-resolution seismic data (labels).
    Returns
    -------
    dict[str, object]
        Dictionary containing statistics for low-res data and high-res labels.
    """
    return {
        "low_res": _default_stats(data_low_res),
        "high_res": _default_stats(data_high_res),
    }


def load_files(image_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a seismic files into a numpy array.

    Parameters
    ----------
    image_dir : Path
        Path to folder with the ``.dat`` file.
    shape : tuple[int, int]
        Shape of the array to be loaded.

    Returns
    -------
    np.ndarray
        Loaded array.
    """
    image_dir = Path(image_dir)
    dat_files = sorted(list(image_dir.glob("*.dat")))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {image_dir}")
    data = []
    for dat_file in dat_files:
        arr = np.fromfile(dat_file, dtype=np.float32).reshape(shape).transpose(1, 0)
        data.append(arr)

    data = np.stack(data, axis=0)
    print(f"Loaded {len(data)} files from {image_dir} with shape {shape}.")
    return data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create processed super-resolution dataset for seismic data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing sx and nx2 subfolders with .dat files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where processed_data will be written.",
    )
    parser.add_argument(
        "--partition_method",
        type=str,
        choices=["random"],
        default="random",
        help="Partitioning strategy.",
    )
    parser.add_argument(
        "--percent_val",
        type=float,
        default=10.0,
        help="Percentage of validation samples.",
    )
    parser.add_argument(
        "--percent_test",
        type=float,
        default=20.0,
        help="Percentage of test samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full seismic super-resolution dataset creation workflow."""
    args = parse_args()
    print(f"Using arguments: {args}")

    partition_name = f"partition_method_{args.partition_method}"
    output_partition_dir = Path(args.output_dir) / partition_name
    try:
        output_partition_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Output directory {output_partition_dir} already exists. "
            "Remove it or choose another output path."
        ) from exc

    print("Step 1/4 - Load files from input directory into a numpy array")
    data_dir = Path(args.data_dir)
    high_res_dir = data_dir / "sx"
    low_res_dir = data_dir / "nx2"
    if not high_res_dir.is_dir() or not low_res_dir.is_dir():
        raise FileNotFoundError(
            f"Expected subdirectories 'sx' and 'nx2' in {data_dir} not found."
        )

    # Load high-res and low-res data. The shape is (num_samples, height, width).
    low_res_data = load_files(low_res_dir, shape=(128, 128))
    high_res_data = load_files(high_res_dir, shape=(256, 256))

    if len(low_res_data) != len(high_res_data):
        raise ValueError(
            f"Number of low-res samples ({len(low_res_data)}) does not match number of high-res samples ({len(high_res_data)})."
        )


    print("Step 2/4 - Split data into low-res and high-res pairs")
    train_indices, val_indices, test_indices = random_split(
        num_samples=low_res_data.shape[0],
        percent_val=args.percent_val,
        percent_test=args.percent_test,
        seed=args.seed,
    )

    train_high_res = high_res_data[train_indices]
    train_low_res = low_res_data[train_indices]
    val_high_res = high_res_data[val_indices]
    val_low_res = low_res_data[val_indices]
    test_high_res = high_res_data[test_indices]
    test_low_res = low_res_data[test_indices]

    print("Step 3/4 - Calculate statistics")
    stats: dict[str, object] = {
        "partition_method": args.partition_method,
        "split_train": get_statistics(train_high_res, train_low_res),
        "split_val": get_statistics(val_high_res, val_low_res),
        "split_test": get_statistics(test_high_res, test_low_res),
    }

    print("Step 4/4 - Save train/val/test samples")
    for split_name, indices, low_res_split, high_res_split in [
        ("train", train_indices, train_low_res, train_high_res),
        ("val", val_indices, val_low_res, val_high_res),
        ("test", test_indices, test_low_res, test_high_res),
    ]:
        split_dir = output_partition_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=False)
        split_data_dir = split_dir / "data"
        split_label_dir = split_dir / "label"
        split_data_dir.mkdir(parents=True, exist_ok=False)
        split_label_dir.mkdir(parents=True, exist_ok=False)

        # Iterate over samples in the split and save each low-res/high-res pair as .npy files (data/label subfolders)
        # Use the indices as file names to maintain correspondence between low-res and high-res samples
        for i, idx in enumerate(indices):
            low_res_sample = low_res_data[idx]
            high_res_sample = high_res_data[idx]
            np.save(split_data_dir / f"{idx:04d}.npy", low_res_sample)
            np.save(split_label_dir / f"{idx:04d}.npy", high_res_sample)

        print(f"Saved {len(indices)} samples for {split_name} split in {split_dir}")


    stats_output = Path(args.output_dir) / f"{partition_name}_stats.json"
    with open(stats_output, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(stats, indent=4))
    print(f"Saved statistics to {stats_output}")
    print(f"Done. Processed dataset available at {output_partition_dir}")


if __name__ == "__main__":
    main()
