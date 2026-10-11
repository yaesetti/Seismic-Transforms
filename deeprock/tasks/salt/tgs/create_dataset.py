"""Create the processed TGS salt segmentation dataset.

Tutorial
--------
1. Download the TGS dataset and copy train_images/ and train_masks/.
2. Run this script to split files and export samples as .npy.
3. Use plot_dataset.py to inspect generated samples.

Example
-------
python create_dataset.py \
    --data_dir ../../../data/original/tgs \
    --output_dir processed_data \
    --partition_method random \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split


def _default_stats(data: np.ndarray) -> dict[str, float | str | list[int]]:
    """Compute basic descriptive statistics for one array.

    Parameters
    ----------
    data : np.ndarray
        Input array to summarize.

    Returns
    -------
    dict[str, float | str | list[int]]
        Summary dictionary with scalar stats, dtype, and shape.
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


def get_statistics(data: np.ndarray, label: np.ndarray) -> dict[str, dict[str, object]]:
    """Compute statistics for data and binary label tensors.

    Parameters
    ----------
    data : np.ndarray
        Seismic sample tensor.
    label : np.ndarray
        Binary mask tensor aligned with ``data``.

    Returns
    -------
    dict[str, dict[str, object]]
        Dictionary with ``data`` and ``label`` metric groups.
    """
    print("Calculating statistics...", end=" ", flush=True)
    data_stats = _default_stats(data)
    label_stats = _default_stats(label)

    unique_labels = np.unique(label, return_counts=True)
    label_stats["num_labels"] = int(len(unique_labels[0]))
    label_stats["unique_labels"] = dict(
        zip(unique_labels[0].tolist(), unique_labels[1].tolist())
    )
    label_stats["unique_labels_percent"] = {
        k: float(v / np.prod(label.shape))
        for k, v in label_stats["unique_labels"].items()
    }
    print("Done.")
    return {"data": data_stats, "label": label_stats}


def load_label(file_path: Path) -> np.ndarray:
    """Load and binarize one mask image.

    Parameters
    ----------
    file_path : Path
        Path to PNG mask.

    Returns
    -------
    np.ndarray
        Binary mask array of shape ``(101, 101)`` with dtype ``uint8``.
    """
    data = Image.open(file_path)
    array = np.array(data.getdata()).reshape((101, 101)).astype(np.uint8)
    return np.where(array < 128, 0, 1).astype(np.uint8)


def load_data(file_path: Path) -> np.ndarray:
    """Load one seismic PNG as a 2D ``float32`` array.

    Parameters
    ----------
    file_path : Path
        Path to PNG seismic image.

    Returns
    -------
    np.ndarray
        Seismic array of shape ``(101, 101)`` with dtype ``float32``.
    """
    data = Image.open(file_path)
    array = np.array(data.getdata())[:, 0].reshape((101, 101)).astype(np.float32)
    return array


def random_split(
    files: list[Path],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Split file paths into train/validation/test subsets.

    Parameters
    ----------
    files : list[Path]
        Input sample file paths.
    val_ratio : float
        Validation ratio over total data.
    test_ratio : float
        Test ratio over total data.
    seed : int
        Random seed for deterministic splitting.

    Returns
    -------
    dict[str, list[Path]]
        Mapping with ``train``, ``val``, and ``test`` file lists.
    """
    print("Performing random split...", end=" ", flush=True)
    files_temp, files_test = train_test_split(files, test_size=test_ratio, random_state=seed)
    val_size_adjusted = val_ratio / (1 - test_ratio)
    files_train, files_val = train_test_split(
        files_temp,
        test_size=val_size_adjusted,
        random_state=seed,
    )
    print(f"There are {len(files_train)} train, {len(files_val)} val, and {len(files_test)} test samples.")
    print("Done.")
    return {"train": files_train, "val": files_val, "test": files_test}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for TGS dataset generation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Create processed TGS salt dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train_images/ and train_masks/.",
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
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """Run the complete TGS preprocessing workflow.

    Returns
    -------
    None
        Writes processed arrays and statistics to disk.
    """
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

    print("Step 1/4 - Discover source files")
    input_data_dir = Path(args.data_dir) / "images"
    input_label_dir = Path(args.data_dir) / "masks"
    all_files = sorted(input_data_dir.rglob("*.png"))
    print(f"Found {len(all_files)} images.")

    print("Step 2/4 - Split train/val/test")
    splits = random_split(all_files, args.val_ratio, args.test_ratio, args.seed)

    stats: dict[str, object] = {}
    all_data: list[np.ndarray] = []
    all_label: list[np.ndarray] = []

    print("Step 3/4 - Convert PNG samples to NPY")
    for split_name, split_files in splits.items():
        split_dir = output_partition_dir / split_name
        split_data_dir = split_dir / "data"
        split_label_dir = split_dir / "label"
        split_data_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

        split_data: list[np.ndarray] = []
        split_label: list[np.ndarray] = []

        for file_path in tqdm.tqdm(split_files, desc=f"Processing {split_name} files"):
            label_path = input_label_dir / file_path.name
            data = load_data(file_path)
            label = load_label(label_path)

            # Save using the same stem in both folders to keep data/label pairing explicit.
            np.save(split_data_dir / f"{file_path.stem}.npy", data)
            np.save(split_label_dir / f"{file_path.stem}.npy", label)

            split_data.append(data)
            split_label.append(label)
            all_data.append(data)
            all_label.append(label)

        stats[f"split_{split_name}"] = get_statistics(np.array(split_data), np.array(split_label))
        print(f"Saved {len(split_data)} samples to {split_dir}")

    print("Step 4/4 - Save statistics")
    stats["split_all"] = get_statistics(np.array(all_data), np.array(all_label))
    stats_output = Path(args.output_dir) / f"{partition_name}_stats.json"
    with open(stats_output, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(stats, indent=4))
    print(f"Saved statistics to {stats_output}")
    print(f"Done. Processed dataset available at {output_partition_dir}")


if __name__ == "__main__":
    main()
