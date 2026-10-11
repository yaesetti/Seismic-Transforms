"""Plot selected samples and calculated attributes from the processed F3 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_sample_and_attributes(
    sample: np.ndarray,
    attributes: dict[str, np.ndarray],
    output_file: Path,
    sample_name: str,
) -> None:
    """Plot one seismic sample and its selected attribute maps."""
    fig = plt.figure(figsize=(4 * (len(attributes) + 1), 5))

    plt.subplot(1, len(attributes) + 1, 1)
    plt.imshow(sample, cmap="seismic")
    plt.title(f"Data: {sample_name}")
    plt.axis("off")

    for i, (attr_name, attr_data) in enumerate(attributes.items(), start=2):
        plt.subplot(1, len(attributes) + 1, i)
        plt.imshow(attr_data, cmap="gray")
        plt.title(f"{attr_name.capitalize()}: {sample_name}")
        plt.axis("off")

    output_file.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument(
        "--attributes",
        nargs="+",
        choices=["envelope", "semblance"],
        default=["envelope", "semblance"],
        help="List of attributes to plot alongside the original sample.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the plotting pipeline for selected processed samples."""
    args = parse_args()
    split_data_dir = Path(args.processed_dir) / args.split / "data"
    split_attr_dir = Path(args.processed_dir) / args.split / "attributes"

    if not split_data_dir.exists() or not split_attr_dir.exists():
        raise FileNotFoundError(
            f"Could not find split folders: {split_data_dir} and {split_attr_dir}"
        )

    if args.filenames:
        selected_files = [normalize_filename(name) for name in args.filenames]
    else:
        selected_files = [path.name for path in sorted(split_data_dir.glob("*.npy"))[: args.max_samples]]

    if not selected_files:
        raise ValueError(f"No samples found in {split_data_dir}")

    output_dir = Path(args.output_dir)
    print(f"Plotting {len(selected_files)} sample(s) from split '{args.split}'")
    for sample_name in selected_files:
        data_file = split_data_dir / sample_name
        if not data_file.exists():
            print(f"Skipping missing sample: {sample_name}")
            continue

        sample = np.load(data_file)
        attributes_data: dict[str, np.ndarray] = {}
        missing_attribute = False
        for attr in args.attributes:
            attr_file = split_attr_dir / attr / sample_name
            if not attr_file.exists():
                print(f"Skipping sample '{sample_name}': missing {attr_file}")
                missing_attribute = True
                break
            attributes_data[attr] = np.load(attr_file)

        if missing_attribute:
            continue

        output_file = output_dir / f"{args.split}_{sample_name.replace('.npy', '')}.png"
        plot_sample_and_attributes(sample, attributes_data, output_file, sample_name)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
