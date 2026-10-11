from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import inspect
import io
import sys
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from minerva.data.data_modules.base import MinervaDataModule
from minerva.data.datasets.base import SimpleDataset
from minerva.models.nets.image.deeplabv3 import DeepLabV3, DeepLabV3RegressionHead
from minerva.transforms.transform import (
    CastTo,
    PadCrop,
    Repeat,
    TransformPipeline,
    Unsqueeze,
)

from utils.data import NumpyFolderReader, NumpyPatchReader
from utils.loss import BCEDiceLoss, SRCombinedLoss, MSEWithSqueeze
from utils.transforms import InstanceNormalization, BinarizeLabel
from utils.plot import plot_segmentation_data, plot_regression_data
from utils.metrics import (
    compute_binary_segmentation_metrics,
    compute_multiclass_segmentation_metrics,
    compute_regression_metrics,
    search_best_threshold,
    stack_preallocated,
    WithTargets,
)
import torchinfo
from minerva.pipelines.experiment import using_redirected_outputs


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for a task/dataset pair.

    Attributes
    ----------
    task_type : {"multiclass", "binary", "regression"}
            Problem type used to select losses and evaluation metrics.
    num_classes : int
            Number of output classes for segmentation heads.
    default_partition : str
            Default processed-data partition directory.
    data_sort_method : list of str
            Sort strategy used by data readers.
    label_sort_method : list of str
            Sort strategy used by label readers.
    label_subdir : str, default="label"
            Relative path to labels inside each split directory.
    delimiter : str or None, default=None
            Optional delimiter used by filename sorting.
    key_index : int or list of int, default=0
            Filename split indexes used by sorting.
    needs_padcrop : bool, default=False
            Whether train/val samples should be padded/cropped.
    pad_h : int or None, default=None
            Target height for pad/crop transforms.
    pad_w : int or None, default=None
            Target width for pad/crop transforms.
    superres_output_shape : tuple of int or None, default=None
            Optional output shape for super-resolution regression.
    """

    task_type: Literal["multiclass", "binary", "regression"]
    num_classes: int
    default_partition: str
    data_sort_method: list[str]
    label_sort_method: list[str]
    label_subdir: str = "label"
    delimiter: str | None = None
    key_index: int | list[int] = 0
    needs_padcrop: bool = False
    pad_h: int | None = None
    pad_w: int | None = None
    superres_output_shape: tuple[int, int] | None = None
    instance_norm: bool = False
    binarize_label: bool = False


SUPPORTED_MODELS: tuple[str, ...] = ("deeplabv3", "unet")


DATASET_SPECS: dict[tuple[str, str], DatasetSpec] = {
    (
        "channel_identification",
        "cigChannel",
    ): DatasetSpec(
        task_type="binary",
        num_classes=1,
        default_partition="partition_method_random",
        data_sort_method=["text", "numeric"],
        label_sort_method=["text", "numeric"],
        delimiter="_",
        key_index=[0, 1],
        needs_padcrop=False,
        instance_norm=True,
        binarize_label=True,
    ),
    (
        "facies",
        "aicrowd_seamai",
    ): DatasetSpec(
        task_type="multiclass",
        num_classes=6,
        default_partition="partition_method_test_first",
        data_sort_method=["text", "numeric"],
        label_sort_method=["text", "numeric"],
        delimiter="_",
        key_index=[0, 1],
        needs_padcrop=True,
        pad_h=1006,
        pad_w=590,
        instance_norm=True,
    ),
    (
        "faults",
        "thebe",
    ): DatasetSpec(
        task_type="binary",
        num_classes=1,
        default_partition="partition_method_predefined",
        data_sort_method=["text"],
        label_sort_method=["text"],
        instance_norm=True,
    ),
    (
        "salt",
        "tgs",
    ): DatasetSpec(
        task_type="binary",
        num_classes=1,
        default_partition="partition_method_random",
        data_sort_method=["text"],
        label_sort_method=["text"],
        instance_norm=True,
    ),
    (
        "seismic_attribute_calculation",
        "F3",
    ): DatasetSpec(
        task_type="regression",
        num_classes=1,
        default_partition="partition_method_test_predefined_val_last",
        data_sort_method=["text"],
        label_sort_method=["text"],
        label_subdir="attributes/envelope",
        instance_norm=False,
    ),
    (
        "super_resolution",
        "seismic_super_resolution",
    ): DatasetSpec(
        task_type="regression",
        num_classes=1,
        default_partition="partition_method_random",
        data_sort_method=["text"],
        label_sort_method=["text"],
        superres_output_shape=(256, 256),
        instance_norm=False,
    ),
}


<<<<<<<< HEAD:train.py
========
class MSEWithSqueeze(nn.Module):
    """MSE loss that squeezes singleton channel dimensions when present.

    Notes
    -----
    This keeps regression targets compatible when batches come as
    ``(B, 1, H, W)`` while labels may be ``(B, H, W)``.
    """

    def __init__(self) -> None:
        """Create loss module."""

        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE after optional squeeze.

        Parameters
        ----------
        input : torch.Tensor
                Model predictions.
        target : torch.Tensor
                Ground-truth tensor.

        Returns
        -------
        torch.Tensor
                Scalar MSE loss.
        """

        if input.ndim == 4 and input.shape[1] == 1:
            input = input.squeeze(1)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        return self.mse(input, target)


def squeeze_single_channel(x: torch.Tensor) -> torch.Tensor:
    """Ensure a tensor has an explicit singleton channel dimension."""

    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4 and x.shape[1] == 1:
        return x
    if x.ndim == 4:
        raise ValueError(
            f"Expected a single-channel tensor, got shape {tuple(x.shape)}"
        )
    raise ValueError(f"Expected a 3D or 4D tensor, got shape {tuple(x.shape)}")


def normalize_pair_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize prediction/target pairs using a shared per-sample range."""

    pred = squeeze_single_channel(pred)
    target = squeeze_single_channel(target)

    reduce_dims = (1, 2, 3)
    pair_max = torch.maximum(
        pred.amax(dim=reduce_dims, keepdim=True),
        target.amax(dim=reduce_dims, keepdim=True),
    )
    pair_min = torch.minimum(
        pred.amin(dim=reduce_dims, keepdim=True),
        target.amin(dim=reduce_dims, keepdim=True),
    )
    denom = torch.clamp(pair_max - pair_min, min=eps)
    return (pred - pair_min) / denom, (target - pair_min) / denom


def create_gaussian_window(
    window_size: int,
    channel: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create the Gaussian window used for SSIM computations."""

    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size // 2)
    gaussian_1d = torch.exp(-(coords**2) / (2 * (1.5**2)))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    window_2d = torch.outer(gaussian_1d, gaussian_1d).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim_score(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a single-scale SSIM score."""

    x = squeeze_single_channel(x)
    y = squeeze_single_channel(y)
    _, channel, height, width = x.shape
    real_window_size = min(window_size, height, width)
    window = create_gaussian_window(real_window_size, channel, x.device, x.dtype)

    mu_x = F.conv2d(x, window, padding=0, groups=channel)
    mu_y = F.conv2d(y, window, padding=0, groups=channel)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=0, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=0, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=0, groups=channel) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + eps
    )
    return ssim_map.mean()


def msssim_score(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a multi-scale SSIM score."""

    x = squeeze_single_channel(x)
    y = squeeze_single_channel(y)

    weights = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device, dtype=x.dtype
    )
    mssim_values = []
    mcs_values = []

    for _ in range(len(weights)):
        x = squeeze_single_channel(x)
        y = squeeze_single_channel(y)
        _, channel, height, width = x.shape
        real_window_size = min(window_size, height, width)
        window = create_gaussian_window(real_window_size, channel, x.device, x.dtype)

        mu_x = F.conv2d(x, window, padding=0, groups=channel)
        mu_y = F.conv2d(y, window, padding=0, groups=channel)
        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=0, groups=channel) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=0, groups=channel) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=0, groups=channel) - mu_xy

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        mcs = (2 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2 + eps)
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + eps
        )
        mssim_values.append(ssim_map.mean())
        mcs_values.append(mcs.mean())

        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        y = F.avg_pool2d(y, kernel_size=2, stride=2)

    mssim_values = torch.stack(mssim_values)
    mcs_values = torch.stack(mcs_values)
    mssim_values = (mssim_values + 1) / 2
    mcs_values = (mcs_values + 1) / 2

    return torch.prod(
        (mcs_values[:-1] ** weights[:-1]) * (mssim_values[-1] ** weights[-1])
    )


def calc_psnr_like_reference(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: int = 2,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute PSNR with the same border crop used by the reference repo."""

    diff = (pred - target) / data_range
    shave = scale + 6
    if diff.size(-2) <= 2 * shave or diff.size(-1) <= 2 * shave:
        valid = diff
    else:
        valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    return -10.0 * torch.log10(torch.clamp(mse, min=eps))


class SRCombinedLoss(nn.Module):
    """Super-resolution loss: L1 plus MS-SSIM, matching the notebook flow."""

    def __init__(self, msssim_weight: float = 0.6) -> None:
        super().__init__()
        self.msssim_weight = msssim_weight
        self.l1_weight = 1.0 - msssim_weight
        self.l1 = nn.L1Loss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm, target_norm = normalize_pair_per_sample(input, target)
        l1_term = self.l1(pred_norm, target_norm)
        msssim_term = 1.0 - msssim_score(pred_norm, target_norm, data_range=1.0)
        return self.l1_weight * l1_term + self.msssim_weight * msssim_term


>>>>>>>> benchmark-lib:deeprock/utils/train.py
def parse_devices(raw: str) -> int | list[int]:
    """Parse a device argument into Lightning-compatible format.

    Parameters
    ----------
    raw : str
            Either a single integer (e.g., ``"1"``) or comma-separated device ids
            (e.g., ``"0,1"``).

    Returns
    -------
    int or list of int
            Parsed ``devices`` value for ``lightning.Trainer``.
    """

    if "," not in raw:
        return int(raw)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_trainer_devices(raw: str | int | list[int]) -> str | int | list[int]:
    """Normalize devices for ``lightning.Trainer``.

    Parameters
    ----------
    raw : str or int or list of int
        Input value from CLI or loaded hparams.

    Returns
    -------
    str or int or list of int
        Trainer-compatible devices value.
    """

    if isinstance(raw, list):
        return raw
    if isinstance(raw, int):
        return raw

    value = str(raw).strip()
    if value == "auto":
        return "auto"
    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
        if isinstance(parsed, int):
            return parsed
        return parsed
    if "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    try:
        return int(value)
    except ValueError:
        return value


def parse_precision(raw: str | None) -> str | None:
    """Normalize precision for Lightning Trainer.

    Parameters
    ----------
    raw : str or None
        Precision value.

    Returns
    -------
    str or None
        Precision string or ``None``.
    """

    if raw is None:
        return None
    value = str(raw).strip()
    return None if value == "" else value


def parse_fast_dev_run(raw: Any) -> bool | int:
    """Normalize fast_dev_run for Lightning Trainer.

    Parameters
    ----------
    raw : Any
        Value from CLI or loaded hparams.

    Returns
    -------
    bool or int
        ``False`` when disabled, otherwise ``True`` or an integer batch limit.
    """

    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw

    value = str(raw).strip().lower()
    if value in {"", "false", "0", "none"}:
        return False
    if value in {"true", "1"}:
        return True
    return int(value)


def _serialize_for_yaml(value: Any) -> Any:
    """Convert values to YAML-serializable representations.

    Parameters
    ----------
    value : Any
        Value to serialize.

    Returns
    -------
    Any
        YAML-friendly value.
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize_for_yaml(v) for v in value]
    if isinstance(value, list):
        return [_serialize_for_yaml(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_for_yaml(v) for k, v in value.items()}
    return value


def _extract_cli_keys(argv: list[str]) -> set[str]:
    """Extract argument keys explicitly provided on the command line.

    Parameters
    ----------
    argv : list of str
        Command-line token list (excluding executable).

    Returns
    -------
    set of str
        Argument destination names explicitly present in CLI.
    """

    keys: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", maxsplit=1)[0].replace("-", "_")
        keys.add(key)
    return keys


def _load_hparams(
    hparams_file: Path,
) -> dict[str, Any]:
    """Load hyperparameters from YAML or JSON.

    Parameters
    ----------
    hparams_file : pathlib.Path
        Path to configuration file.

    Returns
    -------
    dict of str to Any
        Loaded hyperparameter dictionary.
    """

    with hparams_file.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in hparams file: {hparams_file}")
    return loaded


def _merge_hparams_with_cli(
    args: argparse.Namespace,
    cli_keys: set[str],
) -> argparse.Namespace:
    """Merge a saved hparams file into parsed args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    cli_keys : set of str
        Argument keys explicitly provided in CLI.

    Returns
    -------
    argparse.Namespace
        Updated namespace after merging file-based values.

    Notes
    -----
    Values from CLI always take precedence over values from the saved hparams file.
    """

    if args.hparams_file is None:
        return args

    hparams_file = Path(args.hparams_file)
    if not hparams_file.exists():
        raise FileNotFoundError(f"hparams file not found: {hparams_file}")

    loaded = _load_hparams(hparams_file)
    for key, value in loaded.items():
        if not hasattr(args, key):
            continue
        if key in cli_keys:
            continue
        current_value = getattr(args, key)
        if isinstance(current_value, Path) and isinstance(value, str):
            setattr(args, key, Path(value))
        else:
            setattr(args, key, value)
    return args


def _validate_required_args(args: argparse.Namespace) -> None:
    """Validate required runtime arguments after hparams merge.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and merged arguments.
    """

    required = ["model", "task", "dataset", "root_data_dir"]
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing required arguments after merge: {joined}. "
            "Pass them via CLI or --hparams_file."
        )


def _build_logger(
    root_log_dir: Path,
    task: str,
    dataset: str,
    run_id: str,
    label_subdir: str | None = None,
) -> CSVLogger:
    """Create a CSV logger.

    Parameters
    ----------
    root_log_dir : pathlib.Path
        Root logging directory.
    task : str
        Task name.
    dataset : str
        Dataset name.
    run_id : str
        Run identifier used as logger version.

    Returns
    -------
    CSVLogger
        Configured CSV logger.
    """
    name = f"{task}/{dataset}"
    if label_subdir:
        name += f"/{label_subdir}"

    return CSVLogger(
        save_dir=str(root_log_dir),
        name=name,
        version=run_id,
    )


def _save_yaml(path: Path, content: dict[str, Any]) -> None:
    """Save dictionary content to YAML file.

    Parameters
    ----------
    path : pathlib.Path
        Output file path.
    content : dict of str to Any
        Data to dump.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=True)


def _load_model_state(
    model: L.LightningModule,
    checkpoint_path: Path,
) -> None:
    """Load model weights from checkpoint file.

    Parameters
    ----------
    model : lightning.LightningModule
        Model instance to receive weights.
    checkpoint_path : pathlib.Path
        Checkpoint path.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


def reader_for_split(
    root_data_dir: Path,
    split: str,
    spec: DatasetSpec,
    label_subdir: str,
    task: str,
    dataset: str,
    patch_size: tuple[int, int],
    patch_stride: tuple[int, int],
    norm: str | None,
) -> tuple[Any, Any]:
    """Build data and label readers for one split.

    Parameters
    ----------
    root_data_dir : pathlib.Path
            Root directory for the selected processed partition.
    split : str
            One of ``train``, ``val``, or ``test``.
    spec : DatasetSpec
            Dataset specification.
        label_subdir : str
            Relative label folder inside split directories.
    task : str
            Task name.
    dataset : str
            Dataset name.
    patch_size : tuple of int
            Patch size for Thebe test reader.
    patch_stride : tuple of int
            Patch stride for Thebe test reader.
    norm : {"min-max", "z-score"} or None
            Optional per-panel normalization for ``NumpyPatchReader``.

    Returns
    -------
    tuple
            Pair ``(data_reader, label_reader)``.
    """

    data_dir = root_data_dir / split / "data"
    label_dir = root_data_dir / split / Path(label_subdir)

    if task == "faults" and dataset == "thebe":
        is_test = split == "test"
        test_patch_size = patch_size if is_test else None
        test_patch_stride = patch_stride if is_test else None
        data_reader = NumpyPatchReader(
            path=data_dir,
            patch_size=test_patch_size,
            patch_stride=test_patch_stride,
            norm=norm,
        )
        label_reader = NumpyPatchReader(
            path=label_dir,
            patch_size=test_patch_size,
            patch_stride=test_patch_stride,
        )
        return data_reader, label_reader

    data_reader = NumpyFolderReader(
        data_dir,
        sort_method=spec.data_sort_method,
        delimiter=spec.delimiter,
        key_index=spec.key_index,
    )
    label_reader = NumpyFolderReader(
        label_dir,
        sort_method=spec.label_sort_method,
        delimiter=spec.delimiter,
        key_index=spec.key_index,
    )
    return data_reader, label_reader


def make_transforms(
    spec: DatasetSpec,
    split: str,
    seed: int,
) -> tuple[TransformPipeline, TransformPipeline]:
    """Create transform pipelines for data and labels.

    Parameters
    ----------
    spec : DatasetSpec
            Dataset specification.
    split : str
            Split name (``train``, ``val``, ``test``).
    seed : int
            Random seed used by stochastic transforms.

    Returns
    -------
    tuple of TransformPipeline
            Data and label transform pipelines.
    """

    data_ops: list[Any] = [Unsqueeze(0)]
    label_ops: list[Any] = []

    if spec.binarize_label:
        label_ops.append(BinarizeLabel())

    if spec.instance_norm:
        data_ops.append(InstanceNormalization(axis=(1, 2)))

    if spec.needs_padcrop and split in {"train", "val"}:
        data_ops.append(
            PadCrop(
                target_h_size=int(spec.pad_h),
                target_w_size=int(spec.pad_w),
                padding_mode="reflect",
                constant_values=0,
                seed=seed,
            )
        )
        label_ops.extend(
            [
                Unsqueeze(0),
                PadCrop(
                    target_h_size=int(spec.pad_h),
                    target_w_size=int(spec.pad_w),
                    padding_mode="reflect",
                    constant_values=0,
                    seed=seed,
                ),
            ]
        )

    data_ops.extend([Repeat(axis=0, n_repetitions=3), CastTo("float32")])

    if spec.task_type in {"multiclass", "binary"}:
        label_ops.append(CastTo("int64"))
    else:
        label_ops.append(CastTo("float32"))

    return TransformPipeline(data_ops), TransformPipeline(label_ops)


def build_dataset(
    root_data_dir: Path,
    split: str,
    spec: DatasetSpec,
    label_subdir: str,
    seed: int,
    task: str,
    dataset: str,
    patch_size: tuple[int, int],
    patch_stride: tuple[int, int],
    norm: str | None,
) -> SimpleDataset:
    """Build a dataset for one split.

    Parameters
    ----------
    root_data_dir : pathlib.Path
            Root directory for the selected processed partition.
    split : str
            Split name.
    spec : DatasetSpec
            Dataset specification.
        label_subdir : str
            Relative label folder inside split directories.
    seed : int
            Random seed used in data transforms.
    task : str
            Task name.
    dataset : str
            Dataset name.
    patch_size : tuple of int
            Patch size for Thebe test reader.
    patch_stride : tuple of int
            Patch stride for Thebe test reader.
    norm : {"min-max", "z-score"} or None
            Optional normalization for Thebe patch reader.

    Returns
    -------
    SimpleDataset
            Configured split dataset.
    """

    data_reader, label_reader = reader_for_split(
        root_data_dir=root_data_dir,
        split=split,
        spec=spec,
        label_subdir=label_subdir,
        task=task,
        dataset=dataset,
        patch_size=patch_size,
        patch_stride=patch_stride,
        norm=norm,
    )
    data_transforms, label_transforms = make_transforms(
        spec=spec, split=split, seed=seed
    )
    return SimpleDataset(
        readers=[data_reader, label_reader],
        transforms=[data_transforms, label_transforms],
    )


def _build_kwargs_for_constructor(
    model_cls: type, candidate_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Filter kwargs by constructor signature.

    Parameters
    ----------
    model_cls : type
            Model class to inspect.
    candidate_kwargs : dict
            Candidate keyword arguments.

    Returns
    -------
    dict
            Only keyword arguments accepted by ``model_cls.__init__``.
    """

    params = inspect.signature(model_cls.__init__).parameters
    return {k: v for k, v in candidate_kwargs.items() if k in params and v is not None}


def _resolve_model_class(model_name: str) -> type:
    """Resolve a model name to a Minerva model class.

    Parameters
    ----------
    model_name : str
            Model identifier.

    Returns
    -------
    type
            Model class.

    Raises
    ------
    ValueError
            If the model name is unknown.
    ImportError
            If the selected model is unavailable in current Minerva version.
    """

    if model_name == "deeplabv3":
        return DeepLabV3
    if model_name == "unet":
        try:
            from minerva.models.nets.image.unet import UNet
        except ImportError as exc:
            raise ImportError(
                "Model 'unet' is not available in the current minerva-ml installation."
            ) from exc
        return UNet
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(
    model_name: str,
    task_type: Literal["multiclass", "binary", "regression"],
    num_classes: int,
    learning_rate: float,
    pretrained: bool,
    pretrained_backbone_path: str | None,
    freeze_backbone: bool,
    bce_weight: float,
    dice_weight: float,
    superres_output_shape: tuple[int, int] | None,
    linear_head: bool = False
) -> L.LightningModule:
    """Build a Lightning model for the selected task.

    Parameters
    ----------
    model_name : str
            Model identifier.
    task_type : {"multiclass", "binary", "regression"}
            Task type to select loss and optional heads.
    num_classes : int
            Number of output classes.
    learning_rate : float
            Optimizer learning rate.
    pretrained : bool
            Whether pretrained weights should be used.
    pretrained_backbone_path : str or None
            Optional path to pretrained weights.
    freeze_backbone : bool
            Whether to freeze backbone parameters.
    bce_weight : float
            BCE weight for binary BCEDice loss.
    dice_weight : float
            Dice weight for binary BCEDice loss.
    superres_output_shape : tuple of int or None
            Optional output shape for super-resolution.
    linear_head : bool
            Whether to use a linear prediction head instead of the default for the selected model.

    Returns
    -------
    lightning.LightningModule
            Configured Lightning model instance.
    """

    model_cls = _resolve_model_class(model_name)

    if task_type == "multiclass":
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    elif task_type == "binary":
        loss_fn = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    elif superres_output_shape is not None:
        loss_fn = SRCombinedLoss(msssim_weight=0.6)
    else:
        loss_fn = MSEWithSqueeze()

    candidate_kwargs: dict[str, Any] = {
        "num_classes": num_classes,
        "loss_fn": loss_fn,
        "learning_rate": learning_rate,
        "pretrained": pretrained,
        "weights_path": pretrained_backbone_path,
        "freeze_backbone": freeze_backbone,
    }

    if task_type == "regression":
        candidate_kwargs["loss_long"] = False
        candidate_kwargs["output_shape"] = superres_output_shape
        if model_name == "deeplabv3":
            if linear_head:
                candidate_kwargs["pred_head"] = nn.Conv2d(2048, 1, kernel_size=1)
            else:
                candidate_kwargs["pred_head"] = DeepLabV3RegressionHead(
                    in_channels=2048,
                    out_channels=1,
                )

    elif task_type == "binary":
        if model_name == "deeplabv3" and linear_head:
            candidate_kwargs["pred_head"] = nn.Conv2d(2048, 1, kernel_size=1)

    elif task_type == "multiclass":
        if model_name == "deeplabv3" and linear_head:
            candidate_kwargs["pred_head"] = nn.Conv2d(2048, num_classes, kernel_size=1)

    model_kwargs = _build_kwargs_for_constructor(model_cls, candidate_kwargs)
    return model_cls(**model_kwargs)


def to_numpy(tensor: Any) -> np.ndarray:
    """Convert a tensor to a NumPy array on CPU.

    Parameters
    ----------
    tensor : Any
            Input tensor.

    Returns
    -------
    np.ndarray
            NumPy array on CPU.
    """

    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    else:
        return np.asarray(tensor)


def to_2d(arr: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3:
        return arr[0]
    elif arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0, 0]
    else:
        raise ValueError(f"Cannot convert array with shape {arr.shape} to 2D.")


def save_multi_segmentation_plots(
    num_classes: int,
    data_info: Dict[str | int, Any],
    output_dir: Path,
    class_colors=None,
) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)

    for k, v in data_info.items():
        data = to_numpy(to_2d(v["data"]))
        label = to_numpy(to_2d(v["label"]))
        prediction = to_numpy(to_2d(v["prediction"]))
        fig = plot_segmentation_data(
            num_classes=num_classes,
            data=data,
            label=label,
            prediction=prediction,
            class_colors=class_colors,
        )
        fig_path = output_dir / f"{k}.png"
        fig.savefig(fig_path)
        print(f"Figure saved to: {fig_path}")


def save_multi_regression_plots(
    data_info: Dict[str | int, Any],
    output_dir: Path,
    show_difference: bool = True,
) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)

    for k, v in data_info.items():
        data = to_numpy(to_2d(v["data"]))
        label = to_numpy(to_2d(v["label"]))
        prediction = to_numpy(to_2d(v["prediction"]))
        fig = plot_regression_data(
            data=data,
            label=label,
            prediction=prediction,
            show_difference=show_difference,
        )
        fig_path = output_dir / f"{k}.png"
        fig.savefig(fig_path)
        print(f"Figure saved to: {fig_path}")


def evaluate(
    model: L.LightningModule,
    data_module: MinervaDataModule,
    trainer: L.Trainer,
    task_type: Literal["multiclass", "binary", "regression"],
    num_classes: int,
    binary_selection_metric: str,
    threshold: float | None,
    log_dir: Path,
    num_samples=10,
    ckpt_name: str = ""
) -> dict[str, float]:
    """Evaluate trained model on validation and test splits.

    Parameters
    ----------
    model : lightning.LightningModule
            Trained model.
    data_module : MinervaDataModule
            Data module with train/val/test datasets.
    trainer : lightning.Trainer
            Configured Lightning trainer.
    task_type : {"multiclass", "binary", "regression"}
            Task type.
    num_classes : int
            Number of classes for multiclass metrics.
    binary_selection_metric : str
            Validation metric used for threshold search.
    threshold : float or None
            Optional fixed threshold for binary tasks.

    Returns
    -------
    dict of str to float
            Evaluation metrics.
    """
    device = trainer.strategy.root_device.type if trainer.strategy else model.device

    print(
        f"Evaluating model. Device: {device}, task type: {task_type}, num classes: {num_classes}"
    )
    prediction_model = WithTargets(
        return_dict=True, return_y=True, output_device=device
    )(model)

    if task_type == "multiclass":

        def _compute_multiclass_segmentation_metrics(loader, name=None, indices=None):
            results = trainer.predict(prediction_model, dataloaders=loader)
            logits = stack_preallocated(results, key="logits")
            targets = stack_preallocated(results, key="y")
            # Free up memory immediately after stacking to avoid OOM on large datasets
            del results
            gc.collect()
            torch.cuda.empty_cache()

            metrics = compute_multiclass_segmentation_metrics(
                logits=logits,
                targets=targets,
                num_classes=num_classes,
                device=str(device),
            )

            if name:
                metrics = {f"{name}/{k}": v for k, v in metrics.items()}

            metrics = {k: float(v) for k, v in metrics.items()}

            if indices is not None:
                logits = to_numpy(logits[indices])
                targets = to_numpy(targets[indices])
                return metrics, logits, targets
            else:
                return metrics

        metrics = {}

        val_indices = np.linspace(
            0, len(data_module.val_dataset) - 1, num=num_samples, dtype=int
        ).tolist()
        val_metrics, val_logits, val_targets = _compute_multiclass_segmentation_metrics(
            data_module.val_dataloader(), name="val", indices=val_indices
        )
        figs_metadata = {
            f"{ckpt_name}_val_{index}": {
                "data": data_module.val_dataset[index][0],
                "label": val_targets[i],
                "prediction": val_logits[i],
            }
            for i, index in enumerate(val_indices)
        }
        save_multi_segmentation_plots(
            num_classes=num_classes,
            data_info=figs_metadata,
            output_dir=log_dir / "prediction_plots",
        )

        test_indices = np.linspace(
            0, len(data_module.test_dataset) - 1, num=num_samples, dtype=int
        ).tolist()
        test_metrics, test_logits, test_targets = (
            _compute_multiclass_segmentation_metrics(
                data_module.test_dataloader(), name="test", indices=test_indices
            )
        )
        figs_metadata = {
            f"{ckpt_name}_test_{index}": {
                "data": data_module.test_dataset[index][0],
                "label": test_targets[i],
                "prediction": test_logits[i],
            }
            for i, index in enumerate(test_indices)
        }
        save_multi_segmentation_plots(
            num_classes=num_classes,
            data_info=figs_metadata,
            output_dir=log_dir / "prediction_plots",
        )

        metrics.update(val_metrics)
        metrics.update(test_metrics)
        return metrics

    elif task_type == "binary":

        def threshold_search(loader):
            results = trainer.predict(prediction_model, dataloaders=loader)
            logits = stack_preallocated(results, key="logits")
            targets = stack_preallocated(results, key="y")
            best_threshold, _ = search_best_threshold(
                logits=logits,
                targets=targets,
                metric_name=binary_selection_metric,
                apply_sigmoid=True,
            )
            return best_threshold

        def _compute_binary_segmentation_metrics(
            loader, threshold, name=None, indices=None
        ):
            results = trainer.predict(prediction_model, dataloaders=loader)
            logits = stack_preallocated(results, key="logits")
            targets = stack_preallocated(results, key="y")
            # Free up memory immediately after stacking to avoid OOM on large datasets
            del results
            gc.collect()
            torch.cuda.empty_cache()

            metrics = compute_binary_segmentation_metrics(
                logits=logits,
                targets=targets,
                threshold=threshold,
                apply_sigmoid=True,
                device=str(device),
            )

            if name:
                metrics = {f"{name}/{k}": v for k, v in metrics.items()}

            metrics = {k: float(v) for k, v in metrics.items()}

            if indices is not None:
                logits = to_numpy(logits[indices])
                targets = to_numpy(targets[indices])
                return metrics, logits, targets
            else:
                return metrics

        if threshold is not None:
            best_threshold = float(threshold)
        else:
            best_threshold = threshold_search(data_module.val_dataloader())

        # Free up memory immediately after threshold search to avoid OOM on large datasets
        gc.collect()
        torch.cuda.empty_cache()

        metrics = {}

        val_indices = np.linspace(
            0, len(data_module.val_dataset) - 1, num=num_samples, dtype=int
        ).tolist()
        val_metrics, val_logits, val_targets = _compute_binary_segmentation_metrics(
            data_module.val_dataloader(),
            threshold=best_threshold,
            name="val",
            indices=val_indices,
        )
        figs_metadata = {
            f"{ckpt_name}_val_{index}": {
                "data": data_module.val_dataset[index][0],
                "label": val_targets[i],
                "prediction": val_logits[i],
            }
            for i, index in enumerate(val_indices)
        }
        save_multi_segmentation_plots(
            num_classes=2,
            data_info=figs_metadata,
            output_dir=log_dir / "prediction_plots",
            class_colors={0: "#000000", 1: "#FFFFFF"},
        )

        test_indices = np.linspace(
            0, len(data_module.test_dataset) - 1, num=num_samples, dtype=int
        ).tolist()
        test_metrics, test_logits, test_targets = _compute_binary_segmentation_metrics(
            data_module.test_dataloader(),
            threshold=best_threshold,
            name="test",
            indices=test_indices,
        )
        figs_metadata = {
            f"{ckpt_name}_test_{index}": {
                "data": data_module.test_dataset[index][0],
                "label": test_targets[i],
                "prediction": test_logits[i],
            }
            for i, index in enumerate(test_indices)
        }
        save_multi_segmentation_plots(
            num_classes=2,
            data_info=figs_metadata,
            output_dir=log_dir / "prediction_plots",
            class_colors={0: "#000000", 1: "#FFFFFF"},
        )

        metrics.update(val_metrics)
        metrics.update(test_metrics)
        metrics["best_threshold"] = float(best_threshold)
        return metrics

    elif task_type == "regression":

        def _compute_regression_metrics(loader, name=None, indices=None):
            results = trainer.predict(prediction_model, dataloaders=loader)
            logits = stack_preallocated(results, key="logits")
            targets = stack_preallocated(results, key="y")
            # Free up memory immediately after stacking to avoid OOM on large datasets
            del results
            gc.collect()
            torch.cuda.empty_cache()

            metrics = compute_regression_metrics(
                logits=logits,
                targets=targets,
                device=str(device),
            )

            if name:
                metrics = {f"{name}/{k}": v for k, v in metrics.items()}

            metrics = {k: float(v) for k, v in metrics.items()}

<<<<<<<< HEAD:train.py
            if indices is not None:
                logits = to_numpy(logits[indices])
                targets = to_numpy(targets[indices])
                return metrics, logits, targets
            else:
                return metrics

        metrics = {}
        val_indices = np.linspace(
            0, len(data_module.val_dataset) - 1, num=num_samples, dtype=int
        ).tolist()
        val_metrics, val_logits, val_targets = _compute_regression_metrics(
            data_module.val_dataloader(), name="val", indices=val_indices
========
    if superres_output_shape is not None:
        val_pred_norm, val_target_norm = normalize_pair_per_sample(
            val_logits, val_targets
        )
        test_pred_norm, test_target_norm = normalize_pair_per_sample(
            test_logits, test_targets
        )
        val_mse = torch.mean(
            (val_pred_norm.float().cpu() - val_target_norm.float().cpu()) ** 2
        )
        test_mse = torch.mean(
            (test_pred_norm.float().cpu() - test_target_norm.float().cpu()) ** 2
        )
        test_mae = torch.mean(
            torch.abs(test_pred_norm.float().cpu() - test_target_norm.float().cpu())
>>>>>>>> benchmark-lib:deeprock/utils/train.py
        )
        figs_metadata = {
            f"{ckpt_name}_val_{index}": {
                "data": data_module.val_dataset[index][0],
                "label": val_targets[i],
                "prediction": val_logits[i],
            }
            for i, index in enumerate(val_indices)
        }
        save_multi_regression_plots(
            data_info=figs_metadata,
            output_dir=log_dir / "prediction_plots",
            show_difference=True,
        )

<<<<<<<< HEAD:train.py
        test_indices = np.linspace(
            0, len(data_module.test_dataset) - 1, num=num_samples, dtype=int
        ).tolist()
        test_metrics, test_logits, test_targets = _compute_regression_metrics(
            data_module.test_dataloader(), name="test", indices=test_indices
        )
        figs_metadata = {
            f"{ckpt_name}_test_{index}": {
                "data": data_module.test_dataset[index][0],
                "label": test_targets[i],
                "prediction": test_logits[i],
            }
            for i, index in enumerate(test_indices)
        }
        save_multi_regression_plots(
            data_info=figs_metadata,
            output_dir=log_dir / "prediction_plots",
            show_difference=True,
        )

        metrics.update(val_metrics)
        metrics.update(test_metrics)
        return metrics

    else:
        raise ValueError(f"Unsupported task type: {task_type}")
========
    val_mse = torch.mean((val_logits.float().cpu() - val_targets.float().cpu()) ** 2)
    test_mse = torch.mean((test_logits.float().cpu() - test_targets.float().cpu()) ** 2)
    test_mae = torch.mean(
        torch.abs(test_logits.float().cpu() - test_targets.float().cpu())
    )
    return {
        "val_mse": float(val_mse.item()),
        "test_mse": float(test_mse.item()),
        "test_mae": float(test_mae.item()),
    }
>>>>>>>> benchmark-lib:deeprock/utils/train.py


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
            Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train and evaluate Lightning segmentation/regression models on seismic datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=list(SUPPORTED_MODELS),
        default=None,
        help="Model architecture to use.",
    )
    parser.add_argument("--task", default=None, help="Task name (e.g. faults, facies).")
    parser.add_argument(
        "--dataset", default=None, help="Dataset name (e.g. thebe, tgs)."
    )

    parser.add_argument(
        "--root_data_dir",
        type=Path,
        default=None,
        help=(
            "Path to processed partition directory that already contains "
            "train/val/test splits."
        ),
    )
    parser.add_argument(
        "--root_log_dir",
        type=Path,
        default=Path("logs"),
        help="Root directory where Lightning CSV logs and checkpoints are stored.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier used as logger version.",
    )
    parser.add_argument(
        "--hparams_file",
        type=Path,
        default=None,
        help="Load training configuration from YAML/JSON hparams file.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Resume training from a Lightning checkpoint path.",
    )
    parser.add_argument(
        "--label_subdir",
        type=str,
        default=None,
        help=(
            "Label subdirectory inside each split. "
            "Defaults to the dataset specification (for example, label or "
            "attributes/envelope)."
        ),
    )
    parser.add_argument(
        "--progress_bar_refresh_rate",
        type=int,
        default=1,
        help=(
            "How often to refresh the progress bar (in steps). "
            "Set to 0 to disable progress bar updates."
        ),
    )
    parser.add_argument(
        "--step",
        choices=["train", "evaluate", "train-evaluate"],
        default="train-evaluate",
        help="Execution step to run.",
    )

    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained backbone."
    )
    parser.add_argument(
        "--pretrained_backbone_path",
        type=str,
        default=None,
        help="Path to pretrained DeepLabV3/backbone weights.",
    )
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--devices",
        default="auto",
        help="Lightning Trainer devices value (str, int, or list-style comma string).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Lightning Trainer precision value.",
    )
    parser.add_argument(
        "--fast_dev_run",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Lightning Trainer fast_dev_run value. Pass without a value to enable, "
            "or supply an integer batch limit."
        ),
    )
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--limit_test_batches", type=float, default=1.0)
    parser.add_argument("--enable_checkpointing", action="store_true")
    parser.add_argument("--disable_logger", action="store_true")

    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument(
        "--binary_selection_metric",
        choices=["dice", "iou", "jaccard"],
        default="iou",
    )
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument(
        "--patch_size",
        nargs=2,
        type=int,
        default=[96, 96],
        help="Patch size for Thebe test split reader.",
    )
    parser.add_argument(
        "--patch_stride",
        nargs=2,
        type=int,
        default=[48, 48],
        help="Patch stride for Thebe test split reader.",
    )
    parser.add_argument(
        "--norm",
        choices=["min-max", "z-score"],
        default=None,
        help="Optional patch normalization for Thebe data reader.",
    )

    parser.add_argument(
        "--results_output",
        type=Path,
        default=None,
        help="Optional YAML file to save evaluation results.",
    )
    parser.add_argument(
        "--linear_head",
        action="store_true",
        help=(
            "Use a linear prediction head instead of the default for the selected model. "
            "For DeepLabV3, the linear head consists of adaptive average pooling followed by a fully connected layer. "
        ),
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    """Run CLI workflow: parse args, train, evaluate, and optionally save metrics."""

    args = parse_args()
    cli_keys = _extract_cli_keys(sys.argv[1:])
    args = _merge_hparams_with_cli(args=args, cli_keys=cli_keys)
    _validate_required_args(args)

    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    spec_key = (args.task, args.dataset)
    if spec_key not in DATASET_SPECS:
        available = ", ".join([f"{task}/{dataset}" for task, dataset in DATASET_SPECS])
        raise ValueError(
            f"Unsupported task/dataset '{args.task}/{args.dataset}'. Available: {available}"
        )

    spec = DATASET_SPECS[spec_key]
    root_data_dir = Path(args.root_data_dir)
    label_subdir = args.label_subdir or spec.label_subdir
    if label_subdir is not None:
        label_subdir = Path(label_subdir)

    if not root_data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {root_data_dir}")

    patch_size = tuple(args.patch_size)
    patch_stride = tuple(args.patch_stride)
    pretrained = args.pretrained or args.pretrained_backbone_path is not None
    devices = parse_trainer_devices(args.devices)
    precision = parse_precision(args.precision)
    fast_dev_run = parse_fast_dev_run(args.fast_dev_run)
    evaluate_only = args.step == "evaluate"

    if evaluate_only:
        logger: CSVLogger | bool = False
    elif fast_dev_run is False and args.disable_logger:
        logger: CSVLogger | bool = False
    elif fast_dev_run is False:
        logger = _build_logger(
            root_log_dir=args.root_log_dir,
            task=args.task,
            dataset=args.dataset,
            run_id=str(args.run_id),
            label_subdir=args.label_subdir,
        )
    else:
        logger = False

    log_dir = (
        Path(logger.log_dir)
        if isinstance(logger, CSVLogger)
        else args.root_log_dir / args.task / args.dataset / str(args.run_id)
    )

    if fast_dev_run is False and not evaluate_only:
        if log_dir.exists():
            print(f"Removing existing log directory: {log_dir}")
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline() -> dict[str, dict[str, float]]:
        print(f"Training model '{args.model}' on task '{args.task}/{args.dataset}'")
        if fast_dev_run is False and not evaluate_only:
            print(f"Log directory: {log_dir}")

        L.seed_everything(args.seed, workers=True)

        train_dataset = build_dataset(
            root_data_dir=root_data_dir,
            split="train",
            spec=spec,
            label_subdir=label_subdir,
            seed=args.seed,
            task=args.task,
            dataset=args.dataset,
            patch_size=patch_size,
            patch_stride=patch_stride,
            norm=args.norm,
        )
        val_dataset = build_dataset(
            root_data_dir=root_data_dir,
            split="val",
            spec=spec,
            label_subdir=label_subdir,
            seed=args.seed,
            task=args.task,
            dataset=args.dataset,
            patch_size=patch_size,
            patch_stride=patch_stride,
            norm=args.norm,
        )
        test_dataset = build_dataset(
            root_data_dir=root_data_dir,
            split="test",
            spec=spec,
            label_subdir=label_subdir,
            seed=args.seed,
            task=args.task,
            dataset=args.dataset,
            patch_size=patch_size,
            patch_stride=patch_stride,
            norm=args.norm,
        )

        data_module = MinervaDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            name=f"{args.task}/{args.dataset}",
        )

        model = build_model(
            model_name=args.model,
            task_type=spec.task_type,
            num_classes=spec.num_classes,
            learning_rate=args.learning_rate,
            pretrained=pretrained,
            pretrained_backbone_path=args.pretrained_backbone_path,
            freeze_backbone=args.freeze_backbone,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            superres_output_shape=spec.superres_output_shape,
            linear_head=args.linear_head,
        )

        callbacks: list[Any] = []
        enable_progress_bar = (
            args.progress_bar_refresh_rate > 0 and fast_dev_run is False
        )
        if enable_progress_bar:
            callbacks.append(
                TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)
            )

        best_checkpoint: ModelCheckpoint | None = None
        last_checkpoint: ModelCheckpoint | None = None
        if args.enable_checkpointing and fast_dev_run is False and not evaluate_only:
            checkpoint_dir = log_dir / "checkpoints"
            best_checkpoint = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="best-val-loss-{epoch:03d}-{val_loss:.6f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                save_weights_only=True,
                auto_insert_metric_name=False,
                verbose=True,
            )
            last_checkpoint = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="epoch-{epoch:03d}",
                save_top_k=0,
                save_last=True,
                save_weights_only=False,
                every_n_epochs=1,
            )
            callbacks.extend([best_checkpoint, last_checkpoint])

        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            logger=logger,
            enable_checkpointing=args.enable_checkpointing and fast_dev_run is False,
            callbacks=callbacks,
            accelerator=args.accelerator,
            devices=devices,
            precision=precision,
            fast_dev_run=fast_dev_run,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            limit_test_batches=args.limit_test_batches,
        )

        print("-" * 80)
        print(data_module)
        print("-" * 80)
        print()

        print("-" * 80)
        sample_x, _ = train_dataset[0]
        input_size = (args.batch_size, *tuple(sample_x.shape))
        summary = torchinfo.summary(
            model,
            input_size=input_size,
            verbose=0,
            col_names=("input_size", "output_size", "num_params", "trainable"),
            depth=3,
        )
        print(summary)
        print("-" * 80)
        print()

        print("-" * 80)
        print("Trainer configuration:")
        trainer_params = {
            "max_epochs": args.max_epochs,
            "logger": (not evaluate_only) and (not args.disable_logger),
            "enable_checkpointing": args.enable_checkpointing
            and fast_dev_run is False
            and not evaluate_only,
            "accelerator": args.accelerator,
            "devices": devices,
            "precision": precision,
            "fast_dev_run": fast_dev_run,
            "limit_train_batches": args.limit_train_batches,
            "limit_val_batches": args.limit_val_batches,
            "limit_test_batches": args.limit_test_batches,
        }
        for k, v in trainer_params.items():
            print(f"  {k}: {v}")
        print("-" * 80)
        print()

        if fast_dev_run is False and not evaluate_only:
            all_hparams = {k: _serialize_for_yaml(v) for k, v in vars(args).items()}
            _save_yaml(log_dir / "hparams.yaml", all_hparams)
            if isinstance(logger, CSVLogger):
                logger.log_hyperparams(all_hparams)

        results: dict[str, dict[str, float]] = {}

        if args.step in {"train", "train-evaluate"}:
            print("Starting training...")
            trainer.fit(
                model,
                data_module,
                ckpt_path=(
                    str(args.resume_from_checkpoint)
                    if args.resume_from_checkpoint is not None
                    else None
                ),
            )

        if args.step in {"evaluate", "train-evaluate"}:
            if args.step == "evaluate":
                if args.resume_from_checkpoint is None:
                    raise ValueError(
                        "--resume_from_checkpoint is required when --step evaluate is used."
                    )
                _load_model_state(model, Path(args.resume_from_checkpoint))

            print("Starting evaluation...")

            final_results = evaluate(
                model=model,
                data_module=data_module,
                trainer=trainer,
                task_type=spec.task_type,
                num_classes=spec.num_classes,
                binary_selection_metric=args.binary_selection_metric,
                threshold=args.threshold,
                log_dir=log_dir,
                num_samples=10,
                ckpt_name="final_model"
            )
            results["final_model"] = final_results

            if (
                args.step == "train-evaluate"
                and args.enable_checkpointing
                and fast_dev_run is False
                and not evaluate_only
                and best_checkpoint is not None
                and last_checkpoint is not None
            ):
                checkpoint_paths: dict[str, str] = {
                    "best_val_loss": best_checkpoint.best_model_path,
                    "last": last_checkpoint.last_model_path,
                }
                for checkpoint_name, checkpoint_path in checkpoint_paths.items():
                    if not checkpoint_path:
                        continue
                    checkpoint_file = Path(checkpoint_path)
                    if not checkpoint_file.exists():
                        continue

                    checkpoint_model = build_model(
                        model_name=args.model,
                        task_type=spec.task_type,
                        num_classes=spec.num_classes,
                        learning_rate=args.learning_rate,
                        pretrained=pretrained,
                        pretrained_backbone_path=args.pretrained_backbone_path,
                        freeze_backbone=args.freeze_backbone,
                        bce_weight=args.bce_weight,
                        dice_weight=args.dice_weight,
                        superres_output_shape=spec.superres_output_shape,
                        linear_head=args.linear_head,
                    )
                    _load_model_state(checkpoint_model, checkpoint_file)
                    checkpoint_results = evaluate(
                        model=checkpoint_model,
                        data_module=data_module,
                        trainer=trainer,
                        task_type=spec.task_type,
                        num_classes=spec.num_classes,
                        binary_selection_metric=args.binary_selection_metric,
                        threshold=args.threshold,
                        log_dir=log_dir,
                        num_samples=10,
                        ckpt_name=checkpoint_name
                    )
                    results[f"checkpoint_{checkpoint_name}"] = checkpoint_results

        if (
            fast_dev_run is False
            and args.step in {"evaluate", "train-evaluate"}
<<<<<<<< HEAD:train.py
========
            and not evaluate_only
>>>>>>>> benchmark-lib:deeprock/utils/train.py
        ):
            print("\nEvaluation Results")
            for model_name, model_results in results.items():
                print(f"\n[{model_name}]")
                for key, value in model_results.items():
                    print(f"{key:20s}: {value:.6f}")

            results_yaml = log_dir / "results.yaml"
            _save_yaml(
                results_yaml, {k: _serialize_for_yaml(v) for k, v in results.items()}
            )
            print(f"\nSaved results to: {results_yaml}")

            if args.results_output is not None:
                args.results_output.parent.mkdir(parents=True, exist_ok=True)
                _save_yaml(
                    args.results_output,
                    {k: _serialize_for_yaml(v) for k, v in results.items()},
                )
                print(f"\nSaved results to: {args.results_output}")

        return results

    if fast_dev_run is False:
        if isinstance(logger, CSVLogger) and not evaluate_only:
            with using_redirected_outputs(
                output_path=log_dir / "stdout.log",
                error_path=log_dir / "stderr.log",
                mode="w",
                ignore=False,
            ):
                run_pipeline()
        else:
            run_pipeline()
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            run_pipeline()


if __name__ == "__main__":
    main()

