"""Plot selected samples from the processed Thebe fault dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import numpy as np


def plot_sample(data: np.ndarray, label: np.ndarray, output_file: Path, title: str) -> None:
    """Plot one seismic patch and its label mask.

    Parameters
    ----------
    data : np.ndarray
        Seismic patch.
    label : np.ndarray
        Fault mask patch.
    output_file : Path
        Output image path.
    title : str
        Title suffix for the two figure panels.

    Returns
    -------
    None
        Saves figure to disk and closes it.
    """
    if "test" in output_file.stem:
        # For test samples, we are plotting the whole panel instead of a patch
        # it is better to place them horizontally.
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im = axes[0].imshow(data, cmap="seismic", norm=CenteredNorm())
    axes[0].set_title(f"Data: {title}")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0])

    im2 = axes[1].imshow(label, cmap="jet", vmin=0, vmax=max(1, int(np.max(label))))
    axes[1].set_title(f"Label: {title}")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)


def normalize_filename(name: str) -> (str, str):
    """Normalize sample filename to include ``.npy`` extension.

    Parameters
    ----------
    name : str
        Input sample name.

    Returns
    -------
    str
        Filename with ``.npy`` extension.
    """
    path = Path(name)
    name = path.stem + ".npy"
    parent = path.parent
    return (parent, name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Thebe plotting.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Plot selected Thebe samples")
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to processed partition directory (e.g., processed_data/partition_method_predefined).",
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
    """Run plotting over selected files from one processed split.

    Returns
    -------
    None
        Produces image files as side effects.
    """
    args = parse_args()
    split_data_dir = Path(args.processed_dir) / args.split / "data"
    split_label_dir = Path(args.processed_dir) / args.split / "label"

    if not split_data_dir.exists() or not split_label_dir.exists():
        raise FileNotFoundError(
            f"Could not find split folders: {split_data_dir} and {split_label_dir}"
        )

    if args.filenames:
        selected_files = [normalize_filename(name) for name in args.filenames]
    else:
        # Stable ordering gives deterministic sample previews.
        selected_files = [(path.parent.name, path.name) for path in sorted(split_data_dir.glob("**/*.npy"))[: args.max_samples]]
    if not selected_files:
        raise ValueError(f"No samples found in {split_data_dir}")

    output_dir = Path(args.output_dir)
    print(f"Plotting {len(selected_files)} sample(s) from split '{args.split}'")
    for panel_name, sample_name in selected_files:
        data_file = split_data_dir / panel_name / sample_name
        label_file = split_label_dir / panel_name / sample_name
        if not data_file.exists() or not label_file.exists():
            print(f"Skipping missing sample pair: {panel_name}/{sample_name}")
            continue

        data = np.load(data_file)
        label = np.load(label_file)
        # Prefix with split to avoid collisions across train/val/test outputs.
        output_file = output_dir / f"{args.split}_{panel_name}_{sample_name.replace('.npy', '')}.png"
        plot_sample(data, label, output_file, f"{panel_name}_{sample_name}")
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
