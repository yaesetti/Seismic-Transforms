"""Create the processed AICrowd SEAM-AI facies dataset.

Tutorial
--------
1. Put original files in the input folder:
   - data_train.npy
   - labels_train.npy
2. Run this script to create train/val/test splits in .npy format.
3. Use plot_dataset.py to visualize generated samples.

Example
-------
python create_dataset.py \
    --data_dir ../../../data/original/aicrowd_seamai \
    --output_dir processed_data \
    --partition_method test_first \
    --n_val_samples 51 \
    --n_test_samples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_STORED_FORMAT = ["Z", "X", "Y"]


def split_data_test_first(
    data: np.ndarray,
    label: np.ndarray,
    n_val_samples: int,
    n_test_samples: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split dataset slices using a fixed test-first strategy.

    Parameters
    ----------
    data : np.ndarray
        Seismic volume slices arranged along the first axis.
    label : np.ndarray
        Label volume slices aligned with ``data``.
    n_val_samples : int
        Number of validation slices immediately after test slices.
    n_test_samples : int
        Number of initial slices assigned to the test split.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping with keys ``train``, ``val``, and ``test``. Each value is a
        tuple ``(split_data, split_label)``.
    """
    print("Splitting data into train/val/test...", end=" ", flush=True)
    test_data = data[:n_test_samples]
    test_label = label[:n_test_samples]

    val_data = data[n_test_samples : n_test_samples + n_val_samples]
    val_label = label[n_test_samples : n_test_samples + n_val_samples]

    train_data = data[n_test_samples + n_val_samples :]
    train_label = label[n_test_samples + n_val_samples :]
    print(f"There are {train_data.shape[0]} train, {val_data.shape[0]} val, and {test_data.shape[0]} test samples.")
    print("Done.")

    return {
        "train": (train_data, train_label),
        "val": (val_data, val_label),
        "test": (test_data, test_label),
    }


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


def get_statistics(data: np.ndarray, label: np.ndarray) -> dict[str, dict[str, object]]:
    """Compute split statistics for seismic samples and corresponding labels.

    Parameters
    ----------
    data : np.ndarray
        Seismic sample tensor.
    label : np.ndarray
        Label tensor aligned with ``data``.

    Returns
    -------
    dict[str, dict[str, object]]
        Dictionary with ``data`` and ``label`` entries containing scalar stats,
        shape, and label-frequency metadata.
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


def load_array(file_path: Path) -> np.ndarray:
    """Load one NumPy array from disk.

    Parameters
    ----------
    file_path : Path
        Path to ``.npy`` file.

    Returns
    -------
    np.ndarray
        Loaded array.
    """
    print(f"Loading {file_path}...", end=" ", flush=True)
    data = np.load(file_path)
    print("Done.")
    return data


def remap_labels(label_array: np.ndarray) -> np.ndarray:
    """Shift facies labels from one-based indexing to zero-based indexing.

    Parameters
    ----------
    label_array : np.ndarray
        Input label array with classes in the range ``1..6``.

    Returns
    -------
    np.ndarray
        Remapped label array with classes in the range ``0..5``.
    """
    print("Remapping labels from 1-6 to 0-5...", end=" ", flush=True)
    remapped = label_array - 1
    print("Done.")
    return remapped


def save_split_data(
    output_partition_dir: Path,
    split_name: str,
    data: np.ndarray,
    label: np.ndarray,
    prefix: str,
) -> None:
    """Persist one split into ``data`` and ``label`` subdirectories.

    Parameters
    ----------
    output_partition_dir : Path
        Root output folder for the selected partition method.
    split_name : str
        Split identifier (for example, ``train``).
    data : np.ndarray
        Split seismic samples.
    label : np.ndarray
        Split labels aligned with ``data``.
    prefix : str
        Filename prefix (for example, ``xl`` or ``il``).

    Returns
    -------
    None
        This function writes files to disk and returns nothing.
    """
    split_dir = output_partition_dir / split_name
    data_dir = split_dir / "data"
    label_dir = split_dir / "label"
    data_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {split_name} samples ({prefix})...", end=" ", flush=True)
    for idx in range(data.shape[0]):
        np.save(data_dir / f"{prefix}_{idx:03d}.npy", data[idx])
        np.save(label_dir / f"{prefix}_{idx:03d}.npy", label[idx])
    print(f"Done. Saved {data.shape[0]} samples.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create processed AICrowd SEAM-AI facies dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing data_train.npy and labels_train.npy.",
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
        choices=["test_first"],
        default="test_first",
        help="Partitioning strategy.",
    )
    parser.add_argument(
        "--n_val_samples",
        type=int,
        default=51,
        help="Number of validation slices (crossline direction).",
    )
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=200,
        help="Number of test slices (crossline direction).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full facies dataset creation workflow.

    Returns
    -------
    None
        Executes side effects (folder creation, file writing, and logs).
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

    print("Step 1/4 - Load raw arrays")
    raw_dir = Path(args.data_dir)
    data = load_array(raw_dir / "data_train.npy")
    label = remap_labels(load_array(raw_dir / "labels_train.npy"))
    print(f"Loaded raw shapes: data={data.shape}, label={label.shape}")

    print("Step 2/4 - Reorder dimensions for storage")
    # Reorder once so each saved slice is indexed along X for downstream readers.
    permute = (1, 0, 2)  # Z, X, Y -> X, Z, Y
    data = np.transpose(data, permute)
    label = np.transpose(label, permute)
    stored_format = [DEFAULT_STORED_FORMAT[i] for i in permute]
    print(f"Stored shapes: data={data.shape}, label={label.shape}, format={stored_format}")

    print("Step 3/4 - Split and calculate statistics")
    stats: dict[str, object] = {
        "partition_method": args.partition_method,
        "stored_format": stored_format,
        "split_all": get_statistics(data, label),
    }
    split_data = split_data_test_first(data, label, args.n_val_samples, args.n_test_samples)

    print("Step 4/4 - Save train/val/test samples")
    for split_name, (split_data_arr, split_label_arr) in split_data.items():
        stats[f"split_{split_name}"] = get_statistics(split_data_arr, split_label_arr)
        save_split_data(output_partition_dir, split_name, split_data_arr, split_label_arr, prefix="xl")

        if split_name == "train":
            # Keep inline views for training as in the original workflow.
            data_il = np.transpose(split_data_arr, (2, 1, 0))
            label_il = np.transpose(split_label_arr, (2, 1, 0))
            print(f"Saving additional inline training views with shape {data_il.shape}")
            save_split_data(output_partition_dir, split_name, data_il, label_il, prefix="il")

    stats_output = Path(args.output_dir) / f"{partition_name}_stats.json"
    with open(stats_output, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(stats, indent=4))
    print(f"Saved statistics to {stats_output}")
    print(f"Done. Processed dataset available at {output_partition_dir}")


if __name__ == "__main__":
    main()
