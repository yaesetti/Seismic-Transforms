import torch
from torchmetrics import Metric

class BinaryTGSMeanIoU(Metric):
    """
    TGS Salt-like metric: mean precision over IoU thresholds.

    1. Binarize predictions using pred_threshold.
    2. Compute per-sample IoU.
    3. For each IoU threshold, check if IoU > threshold.
    4. Average over thresholds and samples.

    Supports ignore_index in target.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        pred_threshold: float = 0.5,
        iou_thresholds=None,
        ignore_index: int | None = None,
        eps: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pred_threshold = pred_threshold
        self.ignore_index = ignore_index
        self.eps = eps

        if iou_thresholds is None:
            iou_thresholds = torch.arange(0.50, 1.00, 0.05)
        elif isinstance(iou_thresholds, (float, int)):
            iou_thresholds = torch.tensor([float(iou_thresholds)])
        else:
            iou_thresholds = torch.tensor(iou_thresholds, dtype=torch.float32)

        self.register_buffer("iou_thresholds", iou_thresholds)

        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        preds:
            Shape (N, H, W) or (N, 1, H, W).
            Can be probabilities in [0, 1], logits, or binary masks.

        target:
            Shape (N, H, W) or (N, 1, H, W).
            Binary labels: 0/1, with optional ignore_index.
        """
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have the same shape, "
                f"got preds={preds.shape}, target={target.shape}"
            )

        if preds.ndim == 3:
            preds = preds.unsqueeze(1)
            target = target.unsqueeze(1)

        if preds.ndim != 4:
            raise ValueError(
                "Expected preds and target with shape (N, H, W) or (N, 1, H, W)"
            )

        n = preds.shape[0]

        # If logits are passed, convert to probabilities.
        if preds.dtype.is_floating_point:
            if preds.min() < 0 or preds.max() > 1:
                preds = preds.sigmoid()

        pred_bin = preds >= self.pred_threshold

        pred_flat = pred_bin.reshape(n, -1)
        target_flat = target.reshape(n, -1)

        if self.ignore_index is not None:
            valid = target_flat != self.ignore_index
        else:
            valid = torch.ones_like(target_flat, dtype=torch.bool)

        target_bin = target_flat == 1

        valid_pixels = valid.sum(dim=1)

        intersection = (pred_flat & target_bin & valid).sum(dim=1).float()
        union = ((pred_flat | target_bin) & valid).sum(dim=1).float()

        # TGS convention:
        # if prediction and target are both empty, IoU = 1.
        iou = torch.where(
            union > 0,
            intersection / (union + self.eps),
            torch.ones_like(union),
        )

        # Ignore samples with no valid pixels.
        valid_samples = valid_pixels > 0
        if valid_samples.sum() == 0:
            return

        iou = iou[valid_samples]

        thresholds = self.iou_thresholds.to(iou.device)

        correct = iou[:, None] > thresholds[None, :]
        sample_scores = correct.float().mean(dim=1)

        self.score_sum += sample_scores.sum()
        self.n_samples += valid_samples.sum()

    def compute(self) -> torch.Tensor:
        return self.score_sum / self.n_samples.clamp_min(1)
