"""Plot selected samples from the processed TGS salt dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path


def plot_sample(
    data: np.ndarray, label: np.ndarray, output_file: Path, title: str
) -> None:
    """Plot one seismic sample and its 4-class mask."""
    # label_map = {
    #     0: "0: Background\n(non-channel)",
    #     1: "1: Meandering\nChannel",
    #     2: "2: Distributary\nChannel\nNetwork",
    #     3: "3: Submarine\nChannel",
    # }

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # --- Data ---
    im0 = axes[0].imshow(data, cmap="seismic")
    axes[0].set_title(f"Data: {title}")
    axes[0].axis("off")

    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Amplitude")

    # --- Label (discrete) ---
    # colors = ["#000000", "#1f77b4", "#ff7f0e", "#2ca02c"]
    colors = ["#000000", "#ffffff"]
    cmap = ListedColormap(colors)

    # bounds = np.arange(-0.5, 4, 1)
    # norm = BoundaryNorm(bounds, cmap.N)
    # im1 = axes[1].imshow(label, cmap=cmap, norm=norm, interpolation="nearest")
    im1 = axes[1].imshow(label, cmap=cmap, interpolation="nearest")

    axes[1].set_title(f"Label: {title}")
    axes[1].axis("on")

    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    # cbar1.set_ticks([0, 1, 2, 3])
    # cbar1.set_ticklabels([label_map[i] for i in range(4)])
    cbar1.set_ticks([0, 1])
    # cbar1.set_ticklabels(["0: Background", "1: Channel"])

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def normalize_filename(name: str) -> str:
    """Normalize a sample identifier to ``.npy`` filename format.

    Parameters
    ----------
    name : str
        Input sample name.

    Returns
    -------
    str
        Normalized filename.
    """
    return name if name.endswith(".npy") else f"{name}.npy"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for TGS sample plotting.

    Returns
    -------
    argparse.Namespace
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description="Plot selected TGS samples")
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to processed partition directory (e.g., processed_data/partition_method_random).",
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
        help="List of sample filenames (with or without .npy). Note that should contain direction prefix (e.g., 'xl/001_002.npy')",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=3,
        help="Use first N files when --filenames is not provided.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select samples instead of using the first N.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/selected_samples",
        help="Directory to write generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    """Run plotting over user-selected or auto-selected processed samples.

    Returns
    -------
    None
        Produces plot files as side effects.
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
        # Sort for deterministic previews when selecting first N samples.
        selected_files = [Path(path.parent.name) / path.name for path in sorted(split_data_dir.rglob("*.npy"))]
        if args.random:
            np.random.shuffle(selected_files)
        selected_files = selected_files[: args.max_samples]

    print(f"Selected files: {selected_files}")

    if not selected_files:
        raise ValueError(f"No samples found in {split_data_dir}")

    output_dir = Path(args.output_dir)
    print(f"Plotting {len(selected_files)} sample(s) from split '{args.split}'")
    for sample_name in selected_files:
        data_file = split_data_dir / sample_name
        label_file = split_label_dir / sample_name
        print(f"Processing sample: {sample_name}. Data file: {data_file}, Label file: {label_file}")

        if not data_file.exists() or not label_file.exists():
            print(f"Skipping missing sample pair: {sample_name}")
            continue

        data = np.load(data_file)
        label = np.load(label_file)
        # Split prefix prevents name collisions when plotting multiple subsets.
        name = str(sample_name).replace(".npy", "").replace("/", "_")

        output_file = output_dir / f"{args.split}_{name}.png"
        plot_sample(data, label, output_file, name)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
