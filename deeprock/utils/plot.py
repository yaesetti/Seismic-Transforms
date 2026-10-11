import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Optional, Tuple, Dict


def plot_segmentation_data(
    num_classes: int,
    data: np.ndarray,
    label: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    per_image_size: Tuple[int, int] = (5, 7),
    label_names: Optional[Dict[int, str]] = None,
    class_colors: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
):
    """Plot data, label, and optional prediction with per-axis colorbars."""


    # --- Layout ---
    ncols = 2 if prediction is None else 3
    figsize = (per_image_size[0] * ncols, per_image_size[1])
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    if ncols == 2:
        axes = np.array(axes)

    # --- Colormap ---
    if class_colors is None:
        base = plt.cm.get_cmap("tab10", num_classes)
        colors = [base(i) for i in range(num_classes)]
    else:
        colors = [class_colors.get(i, "#cccccc") for i in range(num_classes)]

    cmap = ListedColormap(colors)

    bounds = np.arange(-0.5, num_classes + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # --- Tick labels (safe formatting) ---
    tick_labels = []
    for i in range(num_classes):
        if label_names and i in label_names:
            name = label_names[i].replace(" ", "\n")
            tick_labels.append(f"{i}: {name}")
        else:
            tick_labels.append(str(i))

    # --- Data ---
    im0 = axes[0].imshow(data, cmap="seismic")
    axes[0].set_title("Seismic Data")
    axes[0].axis("off")
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # --- Label ---
    if num_classes == 2:
        kwargs = {"cmap": cmap, "vmin": 0, "vmax": 1, "interpolation": "nearest", "norm": None}
    else:
        kwargs = {"cmap": cmap, "vmin": None, "vmax": None, "interpolation": "nearest", "norm": norm}

    im1 = axes[1].imshow(label, **kwargs)
    axes[1].set_title("Ground Truth")
    axes[1].axis("on")

    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_ticks(range(num_classes))
    cbar1.set_ticklabels(tick_labels)

    # --- Prediction ---
    if prediction is not None:
        im2 = axes[2].imshow(prediction, **kwargs)
        axes[2].set_title("Prediction")
        axes[2].axis("on")

        cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cbar2.set_ticks(range(num_classes))
        cbar2.set_ticklabels(tick_labels)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig

def plot_regression_data(
    data: np.ndarray,
    label: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    per_image_size: Tuple[int, int] = (6, 7),
    data_cmap: str = "seismic",
    label_cmap: str = "viridis",
    pred_cmap: Optional[str] = None,
    diff_cmap: str = "coolwarm",
    show_difference: bool = True,
    title: Optional[str] = None,
):
    """
    Plot regression data, label, optional prediction, and optional difference map.

    Panels:
    - 2 panels: data, label
    - 3 panels: + prediction
    - 4 panels: + (prediction - label) if enabled
    """

    # --- Determine layout ---
    ncols = 2
    if prediction is not None:
        ncols += 1
        if show_difference:
            ncols += 1

    figsize = (per_image_size[0] * ncols, per_image_size[1])
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    if ncols == 1:
        axes = [axes]
    else:
        axes = np.array(axes)

    # --- Default prediction cmap ---
    if pred_cmap is None:
        pred_cmap = label_cmap

    idx = 0

    # --- Data ---
    im0 = axes[idx].imshow(data, cmap=data_cmap)
    axes[idx].set_title("Data")
    axes[idx].axis("off")
    cbar0 = plt.colorbar(im0, ax=axes[idx], fraction=0.046, pad=0.04)
    idx += 1

    # --- Label ---
    im1 = axes[idx].imshow(label, cmap=label_cmap)
    axes[idx].set_title("Label")
    axes[idx].axis("on")
    plt.colorbar(im1, ax=axes[idx], fraction=0.046, pad=0.04)
    idx += 1

    # --- Prediction ---
    if prediction is not None:
        im2 = axes[idx].imshow(prediction, cmap=pred_cmap)
        axes[idx].set_title("Prediction")
        axes[idx].axis("on")
        plt.colorbar(im2, ax=axes[idx], fraction=0.046, pad=0.04)
        idx += 1

        # --- Difference ---
        if show_difference:
            diff = prediction - label
            vmax = np.max(np.abs(diff))  # symmetric color scaling
            im3 = axes[idx].imshow(diff, cmap=diff_cmap, vmin=-vmax, vmax=vmax)
            axes[idx].set_title("Prediction - Label")
            axes[idx].axis("on")
            plt.colorbar(im3, ax=axes[idx], fraction=0.046, pad=0.04)
            idx += 1

    # --- Global title ---
    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    # Prevent suptitle overlap
    if title is not None:
        plt.subplots_adjust(top=0.88)

    # plt.close()  # Prevent display in notebooks
    return fig