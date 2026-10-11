"""Plot selected samples and calculated attributes from the processed F3 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_low_and_high_res(
    low_res_sample: np.ndarray,
    high_res_sample: np.ndarray,
    output_file: Path,
    sample_name: str,
) -> None:
    """Plot one seismic sample and its selected attribute maps."""
    fig = plt.figure(figsize=(10, 5))
    plt.suptitle(f"Sample: {sample_name}", fontsize=16)
    plt.subplot(1, 2, 1)
    plt.imshow(low_res_sample, cmap="gray")
    plt.title("Low-Resolution (128x128)")
    plt.axis("on")
    plt.subplot(1, 2, 2)
    plt.imshow(high_res_sample, cmap="gray")
    plt.title("High-Resolution (256x256)")
    plt.axis("on")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)


def normalize_filename(name: str) -> str:
    """Normalize a sample identifier to include ``.npy`` extension."""
    return name if name.endswith(".npy") else f"{name}.npy"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for sample plotting."""
    parser = argparse.ArgumentParser(description="Plot selected F3 samples and attributes")
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to processed partition directory (e.g., processed_data/partition_method_test_predefined_val_last).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Split to visualize.",
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=None,
        help="List of sample filenames (with or without .npy).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=3,
        help="Use first N files when --filenames is not provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/selected_samples",
        help="Directory to write generated figures.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the plotting pipeline for selected processed samples."""
    args = parse_args()
    print(f"Using arguments: {args}")

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    low_res_dir = processed_dir / f"{args.split}" / "data"
    high_res_dir = processed_dir / f"{args.split}" / "label"

    if not low_res_dir.exists() or not high_res_dir.exists():
        raise FileNotFoundError(
            f"Low-res or high-res directory not found: {low_res_dir}, {high_res_dir}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which files to plot
    if args.filenames:
        sample_names = [normalize_filename(name) for name in args.filenames]
    else:
        sample_names = sorted(low_res_dir.glob("*.npy"))[: args.max_samples]
        sample_names = [name.name for name in sample_names]

    for sample_name in sample_names:
        low_res_file = low_res_dir / sample_name
        high_res_file = high_res_dir / sample_name

        if not low_res_file.exists() or not high_res_file.exists():
            print(f"Warning: Sample files not found for {sample_name}, skipping.")
            continue

        low_res_sample = np.load(low_res_file)
        high_res_sample = np.load(high_res_file)

        output_file = output_dir / f"{args.split}_{sample_name.replace('.npy', '.png')}"
        plot_low_and_high_res(low_res_sample, high_res_sample, output_file, sample_name)
        print(f"Saved plot for {sample_name} to {output_file}")



if __name__ == "__main__":
    main()
