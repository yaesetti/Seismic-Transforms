"""Create the processed cigChannel channels dataset.

Tutorial
--------
1. Put original files in the input folder:
   - data/original/cigChannel/Assorted_Channel_Seismic_300-349
   - data/original/cigChannel/Assorted_Channel_Seismic_350-399
   - data/original/cigChannel/Assorted_Channel_Labels_300-349
   - data/original/cigChannel/Assorted_Channel_Labels_350-399
2. Run create_dataset.py to generate train/val/test splits in the output folder.
3. Use plot_dataset.py to visualize generated samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEFAULT_STORED_FORMAT = ["X", "Y", "Z"]


from collections import Counter
import numpy as np


def aggregate_statistics(
    data_cubes: list[np.ndarray],
    label_cubes: list[np.ndarray],
) -> dict[str, dict[str, object]]:
    """
    Compute global statistics across multiple seismic cubes.

    Parameters
    ----------
    data_cubes : list[np.ndarray]
        List of seismic cubes.
    label_cubes : list[np.ndarray]
        List of label cubes.

    Returns
    -------
    dict
        Aggregated statistics.
    """

    print("Calculating aggregated statistics...", end=" ", flush=True)

    # -------------------------
    # DATA accumulators
    # -------------------------
    data_sum = 0.0
    data_sq_sum = 0.0
    data_count = 0

    data_min = np.inf
    data_max = -np.inf

    # -------------------------
    # LABEL accumulators
    # -------------------------
    label_counter = Counter()
    label_count = 0

    for data, label in zip(data_cubes, label_cubes):

        # Ensure float64 accumulation for numerical stability
        data = data.astype(np.float64)

        # ---- data stats ----
        data_sum += data.sum()
        data_sq_sum += np.square(data).sum()
        data_count += data.size

        data_min = min(data_min, data.min())
        data_max = max(data_max, data.max())

        # ---- label stats ----
        unique, counts = np.unique(label, return_counts=True)
        label_counter.update(dict(zip(unique.tolist(), counts.tolist())))

        label_count += label.size

    # -------------------------
    # Finalize data stats
    # -------------------------
    data_mean = data_sum / data_count

    data_var = (data_sq_sum / data_count) - (data_mean**2)
    data_std = np.sqrt(max(data_var, 0.0))

    data_stats = {
        "mean": float(data_mean),
        "std": float(data_std),
        "min": float(data_min),
        "max": float(data_max),
        "num_voxels": int(data_count),
    }

    # -------------------------
    # Finalize label stats
    # -------------------------
    label_stats = {
        "num_labels": len(label_counter),
        "unique_labels": dict(label_counter),
        "unique_labels_percent": {
            k: float(v / label_count) for k, v in label_counter.items()
        },
        "num_voxels": int(label_count),
    }

    print("Done.")

    return {
        "data": data_stats,
        "label": label_stats,
    }


def load_data(data_path: Path, shape=(256, 256, 256), dtype=np.float32) -> np.ndarray:
    print(
        f"Loading .dat files from {data_path} with expected shape {shape}...",
        end=" ",
        flush=True,
    )
    data = {}
    for d_path in data_path.glob("*.dat"):
        d = np.fromfile(d_path, dtype=dtype).reshape(shape)
        data[int(d_path.stem)] = d
    print("Done.")
    return data


# Multi-class labels are encoded as 100, 200, 300, 400. We remap to 0, 1, 2, 3 for simplicity.
def remap_labels(arr: np.ndarray) -> np.ndarray:
    result = arr // 100
    result[arr == 0] = 0
    return result.astype(np.uint8)


# Alternative simpler remapping if we just want binary labels (channel vs non-channel):
# def remap_labels(arr: np.ndarray) -> np.ndarray:
#     return (arr != 0).astype(np.uint8)


def random_split(
    files: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split files into train/validation/test subsets."""
    # print("Performing random split...", end=" ", flush=True)
    files_temp, files_test = train_test_split(
        files, test_size=test_ratio, random_state=seed
    )
    val_size_adjusted = val_ratio / (1 - test_ratio)
    files_train, files_val = train_test_split(
        files_temp,
        test_size=val_size_adjusted,
        random_state=seed,
    )
    print(
        f"There are {len(files_train)} train, {len(files_val)} val, and {len(files_test)} test files."
    )
    # print(f"Selected files (train): {sorted(files_train)}")
    # print(f"Selected files (val): {sorted(files_val)}")
    # print(f"Selected files (test): {sorted(files_test)}")
    # print("Done.")
    return {"train": files_train, "val": files_val, "test": files_test}


def chunked_random_split(
    files: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    chunks=((0, 50), (50, 100), (100, 300), (300, 400)),
) -> dict[str, list[str]]:
    # Partitionate 0-50, 50-100, 100-300 and 300-400 separately, then combine splits to ensure representation across all ranges.
    print("Performing chunked random split...", end=" ", flush=True)
    files_train, files_val, files_test = [], [], []
    for chunk in chunks:
        in_range = lambda f: chunk[0] <= int(f) < chunk[1]
        chunk_files = [f for f in files if in_range(f)]
        print(
            f"Chunk {chunk}: {len(chunk_files)} files (from {chunk[0]} to {chunk[1]})."
        )
        if len(chunk_files) == 0:
            print(f"Warning: No files found in chunk {chunk}. Skipping.")
            continue
        split = random_split(chunk_files, val_ratio, test_ratio, seed)
        files_train.extend(split["train"])
        files_val.extend(split["val"])
        files_test.extend(split["test"])

    print(f"Selected files for train ({len(files_train)}): {sorted(files_train)}")
    print(f"Selected files for val ({len(files_val)}): {sorted(files_val)}")
    print(f"Selected files for test ({len(files_test)}): {sorted(files_test)}")
    return {"train": files_train, "val": files_val, "test": files_test}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create processed cigChannels channel dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing seismic .dat files.",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing label .dat files.",
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
        "--directions",
        type=str,
        nargs="+",
        choices=["xl", "il", "zl"],
        default=["xl"],
        help="Directions to save slices (xl, il, or both).",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation ratio."
    )
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """Run the full cigChannel dataset creation workflow.

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

    print("Step 1 - Load data arrays")
    data_paths = [Path(d) for d in args.data_dir]
    label_paths = [Path(d) for d in args.label_dir]

    for d in data_paths:
        if not d.is_dir():
            raise ValueError(
                f"Data directory {d} does not exist or is not a directory."
            )
    for d in label_paths:
        if not d.is_dir():
            raise ValueError(
                f"Label directory {d} does not exist or is not a directory."
            )

    data, labels = {}, {}
    for d in data_paths:
        data.update(load_data(d, shape=(256, 256, 256), dtype=np.float32))
    for d in label_paths:
        labels.update(load_data(d, shape=(256, 256, 256), dtype=np.uint16))

    # Check if data and label keys match
    if set(data.keys()) != set(labels.keys()):
        raise ValueError(
            f"Data and label keys do not match. "
            f"Data keys: {sorted(data.keys())}, Label keys: {sorted(labels.keys())}"
        )

    print(f"Loaded {len(data)} samples. Keys: {sorted(data.keys())}")

    print("Step 2 - Remap labels")
    for k in sorted(labels.keys()):
        labels[k] = remap_labels(labels[k])

    print("Step 3 - Split train/val/test")
    keys = sorted(list(data.keys()))
    splits = chunked_random_split(keys, args.val_ratio, args.test_ratio, args.seed)

    stats = {"stored_format": DEFAULT_STORED_FORMAT}

    print("Step 4 - Save processed arrays")
    for split_name in ["train", "val", "test"]:
        split_data_dir = output_partition_dir / split_name / "data"
        split_labels_dir = output_partition_dir / split_name / "label"
        split_data_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        tqdm_bar = tqdm(
            splits[split_name],
            desc=f"Saving {split_name} data",
            total=len(splits[split_name]),
        )

        split_data, split_labels = [], []

        for key in tqdm_bar:
            current_data, current_label = data[key], labels[key]
            split_data.append(current_data)
            split_labels.append(current_label)
            for direction in args.directions:
                direction_data_folder = split_data_dir / direction
                direction_data_folder.mkdir(parents=True, exist_ok=True)

                direction_label_folder = split_labels_dir / direction
                direction_label_folder.mkdir(parents=True, exist_ok=True)

                # if direction == "xl", save slices along X dimension [i, :, :];
                # if direction == "il", save slices along Y dimension [:, i, :];
                # if direction == "zl", save slices along Z dimension [:, :, i].
                if direction == "xl":
                    dim_len = current_data.shape[0]
                    slice_fn = lambda i: (i, slice(None), slice(None))
                elif direction == "il":
                    dim_len = current_data.shape[1]
                    slice_fn = lambda i: (slice(None), i, slice(None))
                else:
                    dim_len = current_data.shape[2]
                    slice_fn = lambda i: (slice(None), slice(None), i)

                for i in range(dim_len):
                    tqdm_bar.set_postfix(
                        {
                            "cube": f"{key:03d}",
                            "direction": direction,
                            "slice": f"{i:03d}",
                        }
                    )
                    the_slice = slice_fn(i)
                    # transpose to keep consistent orientation on disk
                    np.save(
                        direction_data_folder / f"{key:03d}_{i:03d}.npy",
                        current_data[the_slice].T,
                    )
                    np.save(
                        direction_label_folder / f"{key:03d}_{i:03d}.npy",
                        current_label[the_slice].T,
                    )

        # Calculcate split statistics
        split_stats = aggregate_statistics(split_data, split_labels)
        stats[f"split_{split_name}"] = split_stats

    # Save overall statistics
    stats_output = Path(args.output_dir) / f"{partition_name}_stats.json"
    with open(stats_output, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(stats, indent=4))
    print(f"Saved statistics to {stats_output}")
    print(f"Done. Processed dataset available at {output_partition_dir}")

    # Create a single numpy array by stacking all samples along a new dimension
    # The data is (N, 256, 256, 256), we flatten first two dimensions to to tuen into:
    # (N*256, 256, 256) so each slice along the first dimension is indexed along X.
    # train_data = np.stack([data[k] for k in splits["train"]], axis=0).reshape(
    #     -1, 256, 256
    # )
    # train_label = np.stack([labels[k] for k in splits["train"]], axis=0).reshape(
    #     -1, 256, 256
    # )
    # val_data = np.stack([data[k] for k in splits["val"]], axis=0).reshape(-1, 256, 256)
    # val_label = np.stack([labels[k] for k in splits["val"]], axis=0).reshape(
    #     -1, 256, 256
    # )
    # test_data = np.stack([data[k] for k in splits["test"]], axis=0).reshape(
    #     -1, 256, 256
    # )
    # test_label = np.stack([labels[k] for k in splits["test"]], axis=0).reshape(
    #     -1, 256, 256
    # )

    # print(
    #     f"Stacked shapes: train_data={train_data.shape}, train_label={train_label.shape}"
    # )
    # print(f"Stacked shapes: val_data={val_data.shape}, val_label={val_label.shape}")
    # print(f"Stacked shapes: test_data={test_data.shape}, test_label={test_label.shape}")

    # print("Step 4 - Save processed arrays")
    # for split_name, (d, l) in zip(
    #     ["train", "val", "test"],
    #     [(train_data, train_label), (val_data, val_label), (test_data, test_label)],
    # ):
    #     split_data_dir = output_partition_dir / split_name / "data"
    #     split_labels_dir = output_partition_dir / split_name / "label"
    #     split_data_dir.mkdir(parents=True, exist_ok=True)
    #     split_labels_dir.mkdir(parents=True, exist_ok=True)

    #     for direction in args.directions:
    #         for i in tqdm(
    #             range(d.shape[0]), desc=f"Saving {split_name} data", total=d.shape[0]
    #         ):
    #             if direction == "xl":
    #                 the_slice = (i, slice(None), slice(None))
    #             elif direction == "il":
    #                 the_slice = (slice(None), i, slice(None))
    #             else:
    #                 the_slice = (slice(None), slice(None), i)

    #             np.save(split_data_dir / f"{direction}_{i:06d}.npy", d[the_slice].T)
    #             np.save(split_labels_dir / f"{direction}_{i:06d}.npy", l[the_slice].T)

    # print("Step 5 - Compute and save statistics")
    # stats = {
    #     "stored_format": DEFAULT_STORED_FORMAT,
    #     "split_train": get_statistics(train_data, train_label),
    #     "split_val": get_statistics(val_data, val_label),
    #     "split_test": get_statistics(test_data, test_label),
    # }
    # stats_output = Path(args.output_dir) / f"{partition_name}_stats.json"
    # with open(stats_output, "w", encoding="utf-8") as handle:
    #     handle.write(json.dumps(stats, indent=4))
    # print(f"Saved statistics to {stats_output}")
    # print(f"Done. Processed dataset available at {output_partition_dir}")


if __name__ == "__main__":
    main()
