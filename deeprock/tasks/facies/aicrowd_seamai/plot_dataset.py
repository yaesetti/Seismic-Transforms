"""Plot selected samples from the processed AICrowd SEAM-AI dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_sample(data: np.ndarray, label: np.ndarray, output_file: Path, title: str) -> None:
    """Plot one data/label pair and save the figure.

    Parameters
    ----------
    data : np.ndarray
        Seismic 2D sample.
    label : np.ndarray
        Segmentation mask for ``data``.
    output_file : Path
        Destination ``.png`` path.
    title : str
        Title suffix used in both panels.

    Returns
    -------
    None
        The function writes an image file and closes the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].imshow(data, cmap="seismic")
    axes[0].set_title(f"Data: {title}")
    axes[0].axis("on")

    im = axes[1].imshow(label, cmap="jet", vmin=0, vmax=max(1, int(np.max(label))))
    axes[1].set_title(f"Label: {title}")
    axes[1].axis("on")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)


def normalize_filename(name: str) -> str:
    """Ensure a filename ends with ``.npy``.

    Parameters
    ----------
    name : str
        Input filename with or without extension.

    Returns
    -------
    str
        Normalized filename.
    """
    return name if name.endswith(".npy") else f"{name}.npy"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for sample plotting.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Plot selected AICrowd SEAM-AI samples")
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to processed partition directory (e.g., processed_data/partition_method_test_first).",
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
    """Run the plotting pipeline for selected processed samples.

    Returns
    -------
    None
        This function performs file I/O and figure generation only.
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
        # Deterministic ordering keeps sample previews reproducible across runs.
        selected_files = [path.name for path in sorted(split_data_dir.glob("*.npy"))[: args.max_samples]]

    if not selected_files:
        raise ValueError(f"No samples found in {split_data_dir}")

    output_dir = Path(args.output_dir)
    print(f"Plotting {len(selected_files)} sample(s) from split '{args.split}'")
    for sample_name in selected_files:
        data_file = split_data_dir / sample_name
        label_file = split_label_dir / sample_name
        if not data_file.exists() or not label_file.exists():
            print(f"Skipping missing sample pair: {sample_name}")
            continue

        data = np.load(data_file)
        label = np.load(label_file)
        # Keep output naming aligned with split + source sample id.
        output_file = output_dir / f"{args.split}_{sample_name.replace('.npy', '')}.png"
        plot_sample(data, label, output_file, sample_name)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
