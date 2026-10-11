"""Create the processed Thebe fault segmentation dataset.

Tutorial
--------
1. Download Thebe and place files under train/val/test as documented in README.md.
2. Run this script to extract 2D patches and save them as .npy files.
3. Use plot_dataset.py to inspect selected saved samples.

Example
-------
python create_dataset.py \
    --data_dir ../../../data/original/thebe \
    --output_dir processed_data \
    --partition_method predefined \
    --patch_size 96 \
    --patch_stride 48 \
    --mask_percentage 0.03
"""


import argparse
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_STORED_FORMAT = ["Y", "X", "Z"]


def _default_stats(data: np.ndarray) -> dict[str, float | str | list[int]]:
    """Compute basic descriptive statistics for one array.

    Parameters
    ----------
    data : np.ndarray
        Input array.

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
    """Compute split statistics for patch tensors.

    Parameters
    ----------
    data : np.ndarray
        Seismic patches.
    label : np.ndarray
        Fault mask patches aligned with ``data``.

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


def load_npz_array(file_path: Path, key: str = "arr_0") -> np.ndarray:
    """Load one array from an ``.npz`` archive.

    Parameters
    ----------
    file_path : Path
        Path to ``.npz`` archive.
    key : str, default="arr_0"
        Array key inside the archive.

    Returns
    -------
    np.ndarray
        Loaded array.
    """
    print(f"Loading {file_path}...", end=" ", flush=True)
    data = np.load(file_path)[key]
    print("Done.")
    return data


def pad_volume_to_multiple(
    data: np.ndarray,
    label: np.ndarray,
    patch_size: int,
    patch_stride: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad data and label volumes to multiples of patch size.

    Padding is centered in both spatial axes and uses zeros for both arrays.

    Parameters
    ----------
    data : np.ndarray
        Seismic volume in ``(sections, height, width)`` layout.
    label : np.ndarray
        Label volume aligned with ``data``.
    patch_size : int
        Patch side length used during extraction.
    split_name : str
        Name of the current data split (e.g., "train", "val", "test").
        If "test", additional padding is applied to ensure at least two predictions
        for each pixel of the valid region.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Padded ``(data, label)`` volumes.
    """
    if data.shape != label.shape:
        raise ValueError("Data and label shapes must match before padding.")

    height, width = data.shape[1], data.shape[2]

    # Compute minimal padding so each spatial axis is divisible by patch_size.
    total_pad_h = (patch_size-height) % patch_stride
    total_pad_w = (patch_size-width) % patch_stride

    # Split each padding amount across both sides to keep content centered.
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    if split_name == "test":
        # Test split is padded differently (additional 'patch_stride' pixels on each side)
        # probably to ensure at least two predictions for each pixel of the whole valid region
        pad_spec = ((0, 0), (pad_top + patch_stride, pad_bottom + patch_stride), (pad_left + patch_stride, pad_right+ patch_stride))
    else:
        pad_spec = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    padded_data = np.pad(data, pad_spec, mode="reflect")
    padded_label = np.pad(label, pad_spec, mode="reflect")
    return padded_data, padded_label

def extract_panels(
    data: np.ndarray,
    label: np.ndarray,
    **kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[int]]]:
    """Extract 2D panels from the original 3D volumes to get the Test Split.


    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[list[int]]]
        Data patches, label patches, and patch ids.
    """
    data_patches: list[np.ndarray] = []
    label_patches: list[np.ndarray] = []
    names: list[list[int]] = []

    for section_idx in range(0, data.shape[0], 5):
        data_patch = data[section_idx, :, :]
        label_patch = label[section_idx, :, :]
        data_patches.append(data_patch)
        label_patches.append(label_patch)
        names.append([section_idx, data.shape[1], data.shape[2]])

    return data_patches, label_patches, names


def extract_patches(
    data: np.ndarray,
    label: np.ndarray,
    patch_size: int,
    patch_stride: int,
    mask_percentage: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[int]]]:
    """Extract fixed-size patch pairs that satisfy mask coverage criteria.

    Parameters
    ----------
    data : np.ndarray
        Input seismic volume in ``(sections, height, width)`` layout.
    label : np.ndarray
        Input label volume in the same layout as ``data``.
    patch_size : int
        Patch side length in pixels.
    patch_stride : int
        Sliding-window stride in pixels.
    mask_percentage : float | None
        Minimum ratio of positive label pixels for patch retention. If ``None``,
        no filtering is applied and all patches are kept.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[list[int]]]
        Data patches, label patches, and patch ids.
    """
    data_patches: list[np.ndarray] = []
    label_patches: list[np.ndarray] = []
    names: list[list[int]] = []
    pixel_th = int(patch_size * patch_size * mask_percentage) if mask_percentage is not None else 0

    # Slide over each 2D section and keep only informative (fault-containing) patches.
    for section_idx in range(data.shape[0]):
        for row in range(0, data.shape[1] - patch_size + 1, patch_stride):
            for col in range(0, data.shape[2] - patch_size + 1, patch_stride):
                data_patch = data[section_idx, row : row + patch_size, col : col + patch_size]
                label_patch = label[section_idx, row : row + patch_size, col : col + patch_size]

                positive_count = np.sum(label_patch > 0)
                if positive_count >= pixel_th:
                    data_patches.append(data_patch)
                    label_patches.append(label_patch)
                    names.append([section_idx, row, col])

    return data_patches, label_patches, names


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Thebe patch generation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description="Create processed Thebe fault dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train/val/test with seis*.npz and fault*.npz files.",
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
        choices=["predefined"],
        default="predefined",
        help="Partitioning strategy.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=96,
        help="Square patch size in pixels.",
    )
    parser.add_argument(
        "--patch_stride",
        type=int,
        default=48,
        help="Patch extraction stride.",
    )
    parser.add_argument(
        "--mask_percentage",
        type=float,
        default=0.03,
        help="Minimum positive-label ratio required to keep a patch.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full Thebe dataset creation pipeline.

    Returns
    -------
    None
        Executes file I/O and prints progress logs.
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

    input_dir = Path(args.data_dir)
    stats: dict[str, object] = {}

    print("Step 1/4 - Iterate predefined splits")
    for split_name in ["train", "val", "test"]:
        split_dir = input_dir / split_name
        output_split_dir = output_partition_dir / split_name
        output_data_dir = output_split_dir / "data"
        output_label_dir = output_split_dir / "label"
        output_data_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        all_data_patches: list[np.ndarray] = []
        all_label_patches: list[np.ndarray] = []
        all_names: list[str] = []

        print(f"Step 2/4 - Load and patchify split '{split_name}'")
        for data_file in sorted(split_dir.rglob("seis*.npz")):
            label_file = Path(str(data_file).replace("seis", "fault"))
            seismic_volume_idx = int(data_file.stem.replace(f"seis{split_name}", ""))
            print(f"Volume pair: {data_file.name} / {label_file.name}")

            data = load_npz_array(data_file).astype(np.float32)
            label = load_npz_array(label_file).astype(np.uint8)

            # Reorder once for consistent downstream indexing and storage format.
            permute = (0, 2, 1)
            data = np.transpose(data, permute)
            label = np.transpose(label, permute)
            stats["stored_format"] = [DEFAULT_STORED_FORMAT[i] for i in permute]
            mi = data.min(axis=(1, 2))
            ma = data.max(axis=(1, 2))
            mean = data.mean(axis=(1, 2))
            std = data.std(axis=(1, 2))
            step = 5 if split_name == "test" else 1
            for panel in range(0, len(data), step):
                split_index = (seismic_volume_idx-1)*100 + panel
                panel_stats = {
                    "split_index": int(split_index),
                    "split": str(split_name),
                    "min": float(mi[panel]),
                    "max": float(ma[panel]),
                    "mean": float(mean[panel]),
                    "std": float(std[panel]),
                    "seismic_volume_idx": int(seismic_volume_idx),
                    "relative_idx": int(panel),
                }
                os.makedirs(output_data_dir / f"{split_index:04d}", exist_ok=True)
                os.makedirs(output_label_dir / f"{split_index:04d}", exist_ok=True)
                with open(output_data_dir / f"{split_index:04d}/.stats.json", "w", encoding="utf-8") as handle:
                    json.dump(panel_stats, handle, indent=4)
                

            # Pad each volume to guarantee full patch coverage at image borders.
            data, label = pad_volume_to_multiple(data, label, args.patch_size, args.patch_stride, split_name)
            extraction_fn = extract_panels if split_name == "test" else extract_patches
            data_patches, label_patches, names = extraction_fn(
                data=data,
                label=label,
                patch_size=args.patch_size,
                patch_stride=args.patch_stride,
                mask_percentage=args.mask_percentage,
            )
            all_data_patches.extend(data_patches)
            all_label_patches.extend(label_patches)
            all_names.extend([[(seismic_volume_idx-1)*100 + name[0], name[1], name[2]] for name in names])
            print(f"Extracted {len(data_patches)} patches from {data_file.name}")
        print(f"Split {split_name} has {len(all_data_patches)} patches in total.")

        print(f"Step 3/4 - Save split '{split_name}'")
        npy_data_patches = np.array(all_data_patches)
        npy_label_patches = np.array(all_label_patches)
        stats[f"split_{split_name}"] = get_statistics(npy_data_patches, npy_label_patches)

        index = []

        for idx, sample_name in enumerate(all_names):
            file_path = os.path.join(f"{sample_name[0]:04d}", f"{sample_name[1]:04d}_{sample_name[2]:04d}.npy")
            np.save(output_data_dir / file_path, npy_data_patches[idx])
            np.save(output_label_dir / file_path, npy_label_patches[idx])
            index.append(file_path)
        with open(output_data_dir / ".index.json", "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=4)
        with open(output_label_dir / ".index.json", "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=4)
        print(f"Saved {len(all_names)} patches to {output_split_dir}")

    print("Step 4/4 - Save statistics")
    stats_output = Path(args.output_dir) / f"{partition_name}_stats.json"
    with open(stats_output, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(stats, indent=4))
    print(f"Saved statistics to {stats_output}")
    print(f"Done. Processed dataset available at {output_partition_dir}")


if __name__ == "__main__":
    main()
