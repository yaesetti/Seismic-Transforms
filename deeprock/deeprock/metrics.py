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


class BinaryODS(Metric):
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        thresholds=None,
        ignore_index: int | None = None,
        eps: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if thresholds is None:
            thresholds = torch.linspace(0.01, 0.99, 99)
        elif isinstance(thresholds, (float, int)):
            thresholds = torch.tensor([float(thresholds)])
        else:
            thresholds = torch.tensor(thresholds, dtype=torch.float32)

        self.register_buffer("thresholds", thresholds)
        self.ignore_index = ignore_index
        self.eps = eps

        n_t = len(thresholds)
        self.add_state("tp", default=torch.zeros(n_t), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(n_t), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(n_t), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have same shape: {preds.shape} vs {target.shape}"
            )

        if preds.ndim == 3:
            preds = preds.unsqueeze(1)
            target = target.unsqueeze(1)

        if preds.ndim != 4:
            raise ValueError("Expected shape (N,H,W) or (N,1,H,W)")

        if preds.dtype.is_floating_point and (preds.min() < 0 or preds.max() > 1):
            preds = preds.sigmoid()

        preds = preds.reshape(-1)
        target = target.reshape(-1)

        if self.ignore_index is not None:
            valid = target != self.ignore_index
            preds = preds[valid]
            target = target[valid]

        target_bin = target == 1
        thresholds = self.thresholds.to(preds.device)

        pred_bin = preds[:, None] >= thresholds[None, :]

        self.tp += (pred_bin & target_bin[:, None]).sum(dim=0).float()
        self.fp += (pred_bin & ~target_bin[:, None]).sum(dim=0).float()
        self.fn += (~pred_bin & target_bin[:, None]).sum(dim=0).float()

    def compute(self) -> torch.Tensor:
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        return f1.max()


class BinaryOIS(Metric):
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        thresholds=None,
        ignore_index: int | None = None,
        eps: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if thresholds is None:
            thresholds = torch.linspace(0.01, 0.99, 99)
        elif isinstance(thresholds, (float, int)):
            thresholds = torch.tensor([float(thresholds)])
        else:
            thresholds = torch.tensor(thresholds, dtype=torch.float32)

        self.register_buffer("thresholds", thresholds)
        self.ignore_index = ignore_index
        self.eps = eps

        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have same shape: {preds.shape} vs {target.shape}"
            )

        if preds.ndim == 3:
            preds = preds.unsqueeze(1)
            target = target.unsqueeze(1)

        if preds.ndim != 4:
            raise ValueError("Expected shape (N,H,W) or (N,1,H,W)")

        if preds.dtype.is_floating_point and (preds.min() < 0 or preds.max() > 1):
            preds = preds.sigmoid()

        n = preds.shape[0]
        preds = preds.reshape(n, -1)
        target = target.reshape(n, -1)

        if self.ignore_index is not None:
            valid = target != self.ignore_index
        else:
            valid = torch.ones_like(target, dtype=torch.bool)

        target_bin = target == 1
        valid_samples = valid.sum(dim=1) > 0

        if valid_samples.sum() == 0:
            return

        preds = preds[valid_samples]
        target_bin = target_bin[valid_samples]
        valid = valid[valid_samples]

        thresholds = self.thresholds.to(preds.device)

        pred_bin = preds[:, :, None] >= thresholds[None, None, :]

        tp = (pred_bin & target_bin[:, :, None] & valid[:, :, None]).sum(dim=1).float()
        fp = (pred_bin & ~target_bin[:, :, None] & valid[:, :, None]).sum(dim=1).float()
        fn = (~pred_bin & target_bin[:, :, None] & valid[:, :, None]).sum(dim=1).float()

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)

        best_f1_per_sample = f1.max(dim=1).values

        self.score_sum += best_f1_per_sample.sum()
        self.n_samples += best_f1_per_sample.numel()

    def compute(self) -> torch.Tensor:
        return self.score_sum / self.n_samples.clamp_min(1)
