import matplotlib.pyplot as plt
import torch
import torchmetrics
import torch

from typing import Optional, Tuple, Dict
import torch
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F


import lightning as L
import types


class WithTargets:
    """
    Decorator-like utility that modifies a LightningModule instance in-place
    to return both predictions and targets during `predict_step`.

    This implementation preserves the original model instance (no
    re-instantiation), including all weights, initialization arguments, and
    custom attributes.

    Parameters
    ----------
    return_dict : bool, optional (default=True)
        If True, returns a dictionary with keys {"logits", "y"}.
        If False, returns a tuple (logits, y).

    Notes
    -----
    - The original `predict_step` method (if present) is preserved and called.
    - If no `predict_step` exists, the model's forward pass (`self(x)`) is used.
    - The modification is applied at the instance level (monkey patching).
    - This is generally safe for `predict_step`, but may not be reliable for
        all Lightning hooks in distributed or compiled settings.

    Examples
    --------
    >>> model = MyLightningModel(...)
    >>> model = WithTargets(return_dict=True)(model)
    >>> trainer.predict(model)
    """

    def __init__(
        self,
        return_dict: bool = True,
        return_y: bool = True,
        output_device: Optional[str] = None,
    ):
        self.return_dict = return_dict
        self.return_y = return_y
        self.output_device = output_device

    def __call__(self, model):
        """
        Apply the modification to a LightningModule instance.

        Parameters
        ----------
        model : lightning.LightningModule
            The model instance to modify.

        Returns
        -------
        lightning.LightningModule
            The same instance with a modified `predict_step`.
        """

        # ---------------------------------------------------------------------
        # Avoid double patching (idempotency)
        # ---------------------------------------------------------------------
        if hasattr(model, "_with_targets_patched"):
            return model

        # ---------------------------------------------------------------------
        # Save original predict_step (if it exists)
        # ---------------------------------------------------------------------
        original_predict_step = getattr(model, "predict_step", None)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            """
            Wrapped predict_step that returns both predictions and targets.

            Parameters
            ----------
            batch : tuple
                Expected format: (x, y)
            batch_idx : int
                Index of the batch
            dataloader_idx : int, optional
                Index of the dataloader (default=0)

            Returns
            -------
            dict or tuple
                {"logits": logits, "y": y} if return_dict=True,
                otherwise (logits, y)
            """

            # Unpack batch
            x, y = batch

            # -------------------------------------------------------------
            # Compute predictions using original behavior if available
            # -------------------------------------------------------------
            if original_predict_step is not None:
                try:
                    logits = original_predict_step(batch, batch_idx, dataloader_idx)
                except Exception:
                    # Fallback to forward pass if original method fails
                    logits = self(x)
            else:
                # No original predict_step → use forward
                logits = self(x)

            if self._with_targets_output_device is not None:
                logits = logits.to(self._with_targets_output_device)
                y = y.to(self._with_targets_output_device)

            # -------------------------------------------------------------
            # Format output
            # -------------------------------------------------------------
            if self._with_targets_return_dict:
                if self._return_y:
                    return {"logits": logits, "y": y}
                else:
                    return {"logits": logits}
            else:
                if self._return_y:
                    return logits, y
                else:
                    return logits

        # ---------------------------------------------------------------------
        # Store configuration on the instance
        # ---------------------------------------------------------------------
        model._with_targets_return_dict = self.return_dict
        model._return_y = self.return_y
        model._with_targets_output_device = self.output_device

        # Mark as patched (prevents double wrapping)
        model._with_targets_patched = True

        # Keep reference to original method (useful for debugging or rollback)
        model._original_predict_step = original_predict_step

        # ---------------------------------------------------------------------
        # Bind new method to the instance
        # ---------------------------------------------------------------------
        model.predict_step = types.MethodType(predict_step, model)

        return model


def stack_preallocated(preds, key, dim=0, requires_grad=False):
    # Get shapes
    sizes = [p[key].shape[dim] for p in preds]
    total = sum(sizes)

    example = preds[0][key]
    out_shape = list(example.shape)
    out_shape[dim] = total

    # Allocate once
    result = torch.empty(
        out_shape,
        dtype=example.dtype,
        device=example.device,
        requires_grad=requires_grad,
    )

    # Fill
    offset = 0
    for p in preds:
        tensor = p[key]
        length = tensor.shape[dim]

        result[offset : offset + length] = tensor
        offset += length

    return result


@torch.no_grad()
def collect_logits_and_targets(
    model, dataloader, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect logits and targets from a dataloader.
    Returns:
        logits:  (N, H, W)
        targets: (N, H, W)
    """
    model.eval()
    all_logits = []
    all_targets = []

    for batch in tqdm.tqdm(
        dataloader,
        desc="Collecting logits and targets",
        unit="batch",
        total=int(len(dataloader)),
    ):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        # If the model returns (B,1,H,W), squeeze channel
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return logits, targets


@torch.no_grad()
def search_best_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor | None = None,
    metric_name: str = "dice",
    apply_sigmoid: bool = True,
) -> Tuple[float, Dict[float, float]]:
    """
    Search best threshold on validation set.

    logits:   (N, H, W)
    targets:  (N, H, W), binary
    """
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    if thresholds is None:
        thresholds = torch.linspace(0.05, 0.95, 19)  # 0.05 to 0.95 with step of 0.05

    if apply_sigmoid:
        probs = torch.sigmoid(logits)
    targets = targets.int()

    scores = {}

    for thr in tqdm.tqdm(
        thresholds, desc="Searching thresholds", unit="threshold", total=len(thresholds)
    ):
        preds = (probs >= thr).int()

        if metric_name.lower() == "dice":
            metric = torchmetrics.classification.BinaryF1Score()
        elif metric_name.lower() in ["iou", "jaccard"]:
            metric = torchmetrics.classification.BinaryJaccardIndex()
        else:
            raise ValueError(f"Unsupported metric_name={metric_name}")

        score = metric(preds.reshape(-1), targets.reshape(-1)).item()
        scores[float(thr)] = score

    best_thr = max(scores, key=scores.get)
    return best_thr, scores


def plot_threshold_search(
    scores: Dict[float, float],
    metric_name: str = "Dice",
    show: bool = True,
):
    thresholds = list(scores.keys())
    values = list(scores.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, values, marker="o")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Validation Threshold Search ({metric_name})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def _squeeze_4d(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze a 4D tensor with a singleton channel dimension to 3D."""
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    return tensor

@torch.no_grad()
def compute_multiclass_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    device: Optional[str] = None,
    do_argmax: bool = True,
) -> Dict[str, float]:
    """
    Compute metrics for multiclass segmentation.

    logits:  (N, C, H, W)
    targets: (N, H, W), integer class labels
    """
    logits = _squeeze_4d(logits)
    targets = _squeeze_4d(targets)

    if logits.ndim != 4:
        raise ValueError(f"Expected logits shape (N, C, H, W), got {logits.shape}")
    if targets.ndim != 3:
        raise ValueError(f"Expected targets shape (N, H, W), got {targets.shape}")

    device = torch.device(device) if device is not None else logits.device

    if do_argmax:
        preds = torch.argmax(logits, dim=1)
    else:
        preds = logits
    targets = targets.long()

    preds = preds.to(device)
    targets = targets.to(device)

    metric_fns = {
        "pixel_accuracy_macro": torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, average="macro"
        ).to(device),
        "miou_macro": torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=num_classes, average="macro"
        ).to(device),
        "miou_weighted": torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=num_classes, average="weighted"
        ).to(device),
        "per_class_iou": torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=num_classes, average=None
        ).to(device),
    }

    metrics = {}
    for metric_name, metric_fn in metric_fns.items():
        print(f"Computing {metric_name}...")
        if metric_name == "per_class_iou":
            per_class_iou = metric_fn(preds, targets)
            for i in range(num_classes):
                metrics[f"{metric_name}/{i}"] = float(per_class_iou[i].item())
        else:
            metrics[metric_name] = metric_fn(preds, targets).item()

    return metrics


@torch.no_grad()
def compute_binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    device: Optional[str] = None,
    apply_sigmoid: bool = True,
    auprc_thresholds: Optional[int] = 100,
    chunk_size: int = 50_000_000,
) -> Dict[str, float]:

    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    device = torch.device(device) if device is not None else logits.device

    logits = logits.to(device)
    targets = targets.to(device).bool()

    if apply_sigmoid:
        probs = torch.sigmoid(logits)
    else:
        probs = logits

    probs = probs.reshape(-1)
    targets = targets.reshape(-1)

    tp = torch.tensor(0, device=device, dtype=torch.float64)
    tn = torch.tensor(0, device=device, dtype=torch.float64)
    fp = torch.tensor(0, device=device, dtype=torch.float64)
    fn = torch.tensor(0, device=device, dtype=torch.float64)

    # Binned AUPRC; validate_args=False avoids torch.unique validation
    auprc_metric = torchmetrics.classification.BinaryAveragePrecision(
        thresholds=auprc_thresholds,
        validate_args=False,
    ).to(device)

    n = targets.numel()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        p = probs[start:end]
        t = targets[start:end]

        pred = p >= threshold

        tp += (pred & t).sum()
        tn += ((~pred) & (~t)).sum()
        fp += (pred & (~t)).sum()
        fn += ((~pred) & t).sum()

        auprc_metric.update(p, t.long())

    eps = 1e-12

    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    dice_f1 = 2 * tp / (2 * tp + fp + fn + eps)
    iou_jaccard = tp / (tp + fp + fn + eps)

    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / (denominator + eps)

    auprc = auprc_metric.compute()

    metrics = {
        "pixel_accuracy": pixel_accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "specificity": specificity.item(),
        "dice_f1": dice_f1.item(),
        "iou_jaccard": iou_jaccard.item(),
        "mcc": mcc.item(),
        "auprc": auprc.item(),
    }

    metrics["balanced_accuracy"] = 0.5 * (
        metrics["recall"] + metrics["specificity"]
    )

    return metrics

def _squeeze_single_channel(x: torch.Tensor) -> torch.Tensor:
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


def _normalize_pair_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize prediction/target pairs using a shared per-sample range."""

    pred = _squeeze_single_channel(pred)
    target = _squeeze_single_channel(target)

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

    x = _squeeze_single_channel(x)
    y = _squeeze_single_channel(y)
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


@torch.no_grad()
def compute_regression_metrics(
    logits: torch.Tensor, targets: torch.Tensor, device: Optional[str] = None
) -> Dict[str, float]:
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    device = torch.device(device) if device is not None else logits.device
    preds = logits.to(device)
    targets = targets.to(device)

    pred_norm, target_norm = _normalize_pair_per_sample(preds, targets)

    print("Computing MAE...")
    mae = torch.mean(torch.abs(preds - targets))
    print("Computing MSE...")
    mse = torch.mean((preds - targets) ** 2)
    print("Computing RMSE...")
    rmse = torch.sqrt(mse)
    print("Computing PSNR...")
    psnr = calc_psnr_like_reference(pred_norm, target_norm, scale=2)
    print("Computing SSIM...")
    ssim = ssim_score(pred_norm, target_norm, data_range=1.0)

    return {
        "mae": mae.item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
        "psnr": psnr.item(),
        "ssim": ssim.item(),
    }


@torch.no_grad()
def plot_auprc_curve(logits: torch.Tensor, targets: torch.Tensor, show: bool = True):
    probs = torch.sigmoid(logits).reshape(-1)
    targets = targets.int().reshape(-1)

    pr_curve = torchmetrics.classification.BinaryPrecisionRecallCurve()
    precision, recall, thresholds = pr_curve(probs, targets)

    ap_metric = torchmetrics.classification.BinaryAveragePrecision()
    auprc = ap_metric(probs, targets).item()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall.cpu().numpy(), precision.cpu().numpy(), label=f"AUPRC = {auprc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
        return None
    return fig
