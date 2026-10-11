"""Create the processed attribute's calculation task with F3 Dataset.

Tutorial
--------
1. Put original files in the input folder F3/data folder, which should contain:
   - test_once/test1_seismic.npy
   - train/data_train.npy
    Note that we will not use labels as we will generate them from the seismic data using attributes such as envelope and semblance. The original labels are not needed for this task, but you can keep them for reference if desired.
2. Run this script to create train/val/test splits in .npy format. The train/data_train.npy file will be split into train, val while the test_once/test1_seismic.npy will be used as the test set. The output will be organized in a folder structure like:
    processed_data/
    ├── partition_method_test_predefined_val_last.json
    └── partition_method_test_predefined_val_last/
        ├── train/
        │   ├── data/
        │   ├── envelope/
        │   └── semblance/
        ├── val/
        │   ├── data/
        │   ├── envelope/
        │   └── semblance/
        └── test/
            ├── data/
            ├── envelope/
            └── semblance/
3. Use plot_dataset.py to visualize generated samples.

Example
-------
python create_dataset.py \
    --data_dir ../../../data/F3/data \
    --output_dir processed_data \
    --partition_method test_first \
    --n_val_samples 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from attributes import attribute_numpy_envelope, attribute_numpy_semblance

DEFAULT_STORED_FORMAT = ["Z", "X", "Y"]
attributes_map = {
    "envelope": attribute_numpy_envelope,
    "semblance": attribute_numpy_semblance,
}


def split_test_predefined_val_last(
    train_data: np.ndarray,
    train_attributes: dict[str, np.ndarray],
    num_val_samples: int,
):
    """Split train data into train and val sets, keeping test data separate.

    For this partition method, we will use the entire test data as the test set and
    split the train data into train and val sets. The last `num_val_samples` slices
    along the X dimension of the train data will be used for validation, while the
    remaining slices will be used for training.

    Parameters
    ----------
    train_data : np.ndarray
        The full training seismic data array.
    train_attributes : dict[str, np.ndarray]
        Dictionary of attribute arrays corresponding to `train_data`.
    num_val_samples : int
        Number of slices to reserve for validation.

    Returns
    -------
    dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]
        A dictionary with keys 'train' and 'val', where each value is a tuple containing
        the split data array and a dictionary of corresponding attribute arrays.
    """
    total_slices = train_data.shape[0]
    if num_val_samples >= total_slices:
        raise ValueError(
            f"Number of validation samples ({num_val_samples}) must be less than total slices ({total_slices})."
        )

    val_data = train_data[-num_val_samples:]
    train_data_split = train_data[:-num_val_samples]

    val_attributes = {
        attr: arr[-num_val_samples:] for attr, arr in train_attributes.items()
    }
    train_attributes_split = {
        attr: arr[:-num_val_samples] for attr, arr in train_attributes.items()
    }

    return {
        "train": (train_data_split, train_attributes_split),
        "val": (val_data, val_attributes),
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


def get_statistics(
    data: np.ndarray, attributes: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    """Compute split statistics for seismic samples and corresponding labels.

    Parameters
    ----------
    data : np.ndarray
        Seismic sample tensor.
    attributes : dict[str, np.ndarray]
        Dictionary of attribute arrays aligned with ``data``.

    Returns
    -------
    dict[str, dict[str, object]]
        Dictionary with ``data`` and ``label`` entries containing scalar stats,
        shape, and label-frequency metadata.
    """
    print("Calculating statistics...", end=" ", flush=True)
    data_stats = _default_stats(data)
    attr_stats = {attr: _default_stats(arr) for attr, arr in attributes.items()}
    data_stats["attributes"] = attr_stats
    print("Done.")
    return data_stats


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


def save_split_data(
    output_partition_dir: Path,
    split_name: str,
    data: np.ndarray,
    attributes: dict[str, np.ndarray],
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
    attributes : dict[str, np.ndarray]
        Dictionary of attribute arrays aligned with ``data``.
    prefix : str
        Filename prefix (for example, ``xl`` or ``il``).

    Returns
    -------
    None
        This function writes files to disk and returns nothing.
    """

    def _save(arr: np.ndarray, subfolder: str) -> None:
        subfolder = Path(subfolder)
        subfolder.mkdir(parents=True, exist_ok=True)
        for idx in range(arr.shape[0]):
            np.save(subfolder / f"{idx:03d}.npy", arr[idx])
        print(f"Saved {arr.shape[0]} samples to {subfolder}.")

    split_dir = output_partition_dir / split_name
    data_dir = split_dir / "data"
    label_dir = split_dir / "attributes"
    data_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {split_name} data to {data_dir} and attributes to {label_dir}...")
    _save(data, data_dir)
    for attr_name, attr_arr in attributes.items():
        _save(attr_arr, label_dir / attr_name)
    print("Done.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create processed F3 dataset for seismic attribute calculation task."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train and test_once subfolders with raw .npy files.",
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
        choices=["test_predefined_val_last"],
        default="test_predefined_val_last",
        help="Partitioning strategy.",
    )
    parser.add_argument(
        "--n_val_samples",
        type=int,
        default=50,
        help="Number of validation slices.",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        choices=["envelope", "semblance"],
        default=["envelope", "semblance"],
        help="List of attributes to calculate and save.",
    )
    return parser.parse_args()


def calculate_attributes(
    data: np.ndarray, attributes: list[str]
) -> dict[str, np.ndarray]:
    print(f"Calculating attributes: {attributes}...", end=" ", flush=True)
    attr_results = {}
    for attr in attributes:
        if attr not in attributes_map:
            raise ValueError(
                f"Attribute '{attr}' is not supported. Choose from {list(attributes_map.keys())}."
            )
        attr_func = attributes_map[attr]
        attr_results[attr] = attr_func(data)
    print("Done.")
    return attr_results


def main() -> None:
    """Run the full F3 dataset creation workflow.

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

    print("Step 1/6 - Load raw arrays and calculating attributes")
    raw_dir = Path(args.data_dir)
    train_data = load_array(raw_dir / "train" / "train_seismic.npy")
    test_data = load_array(raw_dir / "test_once" / "test1_seismic.npy")
    print(
        f"Loaded raw shapes: train_data={train_data.shape}, test_data={test_data.shape}"
    )

    print("Step 2/6 - Calculating attributes for train and test data...")
    train_attributes = calculate_attributes(train_data, args.attributes)
    test_attributes = calculate_attributes(test_data, args.attributes)

    print("Step 3/6 - Reorder dimensions for storage")
    # Reorder once so each saved slice is indexed along X for downstream readers.
    permute = (0, 2, 1)  # Z, X, Y -> X, Z, Y
    train_data = np.transpose(train_data, permute)
    test_data = np.transpose(test_data, permute)
    train_attributes = {attr: np.transpose(arr, permute) for attr, arr in train_attributes.items()}
    test_attributes = {attr: np.transpose(arr, permute) for attr, arr in test_attributes.items()}
    stored_format = [DEFAULT_STORED_FORMAT[i] for i in permute]
    print(
        f"Stored shapes: train_data={train_data.shape}, test_data={test_data.shape}, format={stored_format}"
    )

    print("Step 4/6 - Splitting data and labels into train/val/test sets...")
    # For this partition method, we will use the entire test data as the test set and
    # split the train data into train and val sets.
    split_data = split_test_predefined_val_last(
        train_data, train_attributes, args.n_val_samples
    )
    split_data["test"] = (test_data, test_attributes)

    print("Step 5/6 - Split and calculate statistics")
    stats: dict[str, object] = {
        "partition_method": args.partition_method,
        "stored_format": stored_format,
        "split_train": get_statistics(split_data["train"][0], split_data["train"][1]),
        "split_val": get_statistics(split_data["val"][0], split_data["val"][1]),
        "split_test": get_statistics(split_data["test"][0], split_data["test"][1]),
    }

    print("Step 6/6 - Save train/val/test samples")
    for split_name, (split_data_arr, split_attr_dict) in split_data.items():
        save_split_data(
            output_partition_dir=output_partition_dir,
            split_name=split_name,
            data=split_data_arr,
            attributes=split_attr_dict,
        )

    stats_output = Path(args.output_dir) / f"{partition_name}_stats.json"
    with open(stats_output, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(stats, indent=4))
    print(f"Saved statistics to {stats_output}")
    print(f"Done. Processed dataset available at {output_partition_dir}")


if __name__ == "__main__":
    main()
