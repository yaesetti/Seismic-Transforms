import sys
import os
import json
import importlib
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


import torch
import torch.nn.functional as F
import numpy as np
from minerva.utils.typing import PathLike


from deeprock.model import DeepRockModel

# Thebe BSDS evaluation protocol constants (paper-aligned defaults for
# BSDSInference / bsds_inference below).
#
# The dataloader pads each Thebe panel up to a multiple of the model's patch
# size/stride so it tiles exactly for sliding-window inference: the true,
# unpadded panel is (1537, 3174), but y_hat/y as seen by `_single_step` are
# reconstructed at the padded shape (1680, 3312). THEBE_ORIGINAL_PANEL_SHAPE
# is the *unpadded* shape: bsds_inference center-crops the padded panel down
# to it (removing the dataloader's padding) before applying the
# upper/lower row bounds below, so those bounds are always interpreted in
# unpadded panel coordinates.
THEBE_ORIGINAL_PANEL_SHAPE: Tuple[int, int] = (1537, 3174)
THEBE_BSDS_UPPER_BOUND: int = 800
THEBE_BSDS_LOWER_BOUND: int = 1300
THEBE_BSDS_SCALEFACTOR: float = 3.0
THEBE_BSDS_N_THRESHOLDS: int = 100
THEBE_BSDS_MAX_DIST: float = 0.0075


@dataclass
class BSDSBatchState:
    """One test batch's BSDS boundary counts, already pooled across that
    batch's samples, for accumulating a true epoch-level ODS/OIS across
    every test batch (see `BSDSInference`).

    Summing these across batches gives the same result as pooling every
    sample in the whole test set at once: both the ODS pooling (sum counts
    over samples, then sweep thresholds) and the OIS pooling (sum each
    sample's own best-threshold counts) are plain sums, which are
    associative regardless of how samples are split into batches.
    """

    thresholds: torch.Tensor  # (n_thresholds,)
    count_r: torch.Tensor  # (n_thresholds,) -- summed over this batch's samples
    sum_r: torch.Tensor  # (n_thresholds,)
    count_p: torch.Tensor  # (n_thresholds,)
    sum_p: torch.Tensor  # (n_thresholds,)
    count_r_best: torch.Tensor  # scalar -- summed over each sample's own best threshold
    sum_r_best: torch.Tensor  # scalar
    count_p_best: torch.Tensor  # scalar
    sum_p_best: torch.Tensor  # scalar


def bsds_inference(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    panel_shape: Tuple[int, int] = THEBE_ORIGINAL_PANEL_SHAPE,
    upper_bound: int = THEBE_BSDS_UPPER_BOUND,
    lower_bound: int = THEBE_BSDS_LOWER_BOUND,
    scale_factor: float = THEBE_BSDS_SCALEFACTOR,
    n_thresholds: int = THEBE_BSDS_N_THRESHOLDS,
    max_dist: float = THEBE_BSDS_MAX_DIST,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], BSDSBatchState]:
    """Preprocess a batch of predicted/GT panels and compute BSDS boundary metrics.

    What BSDS measures
    -------------------
    BSDS (the Berkeley Segmentation Dataset boundary benchmark, adapted for
    Thebe fault panels in `deeprock/utils/bsds2d.py`) scores a prediction as
    a set of *boundaries* rather than as pixels. A predicted boundary pixel
    counts as a true positive if it falls within `max_dist` (a distance
    given as a fraction of the image diagonal) of a ground-truth boundary
    pixel; a ground-truth boundary pixel counts as "recovered" if some
    predicted pixel falls within `max_dist` of it. This tolerance means a
    fault predicted a few pixels off from the labeled fault is still
    rewarded, unlike a strict pixelwise Dice/IoU which would count it as a
    total miss.

    Precision/recall are computed over a sweep of `n_thresholds` probability
    thresholds and reduced to three standard BSDS numbers:

    - **ODS** (Optimal Dataset Scale): match/predicted/ground-truth counts
      are pooled across every sample in the batch *before* sweeping
      thresholds, and the single threshold that maximizes the pooled F1 is
      reported. This is one global operating point shared by the batch.
    - **OIS** (Optimal Image Scale): each sample in the batch independently
      picks its own best threshold; the *counts* at each sample's own best
      threshold (not its F1 score) are then pooled across the batch and
      reduced to a single F1. This is the standard BSDS definition, and is
      **not** the same as averaging each sample's best F1 - pooling counts
      first weights samples by how many boundary pixels they contain,
      exactly as the reference dataset-level evaluation does.
    - **AUPRC**: trapezoidal area under the pooled (ODS) precision-recall
      curve, sorted by recall.

    Scope and caveats
    ------------------
    This function itself is **stateless**: the `metrics` dict it returns
    only pools counts across the samples present in the `y_hat`/`y` batch
    it is given. If the batch contains a single sample (the common case
    here, since one test batch is one reconstructed Thebe panel), ODS and
    OIS in that dict reduce to the same threshold and come out numerically
    identical - with only one sample, "pooled over the batch" and "that
    sample's own best" are the same thing, so this is expected, not a bug.

    The returned `BSDSBatchState` exists precisely to get around this: it
    carries this batch's counts in a form that's safe to *sum* across
    batches (summation is associative, so summing per-batch pooled counts
    gives the same result as pooling every sample in the test set at once).
    The caller is expected to accumulate `BSDSBatchState` across every test
    batch and compute the true, dataset-level ODS/OIS/AUPRC once at the end
    of the test epoch - see `BSDSInference.on_test_epoch_end`, which does
    exactly this. That epoch-level reduction is the one comparable to
    `deeprock.utils.bsds2d.evaluate_thebe_bsds_from_model`; the per-batch
    `metrics` dict is only a fast, per-step monitoring proxy.

    Boundary matching here uses a dilation-based tolerance test (a
    predicted pixel is a hit if it falls inside the dilated GT mask, and
    symmetrically for recall). This is a fast, torch-native
    *approximation* of the correspondence-based (bipartite-matching)
    evaluation used by the compiled py-bsds500 backend: dilation allows
    many-to-many matches and can over-count precision/recall relative to
    true one-to-one correspondence. Treat these numbers as a fast
    in-training proxy for monitoring, not as paper-comparable results.

    Preprocessing (matches the Thebe BSDS paper protocol)
    -------------------------------------------------------
    1. `y_hat` is sigmoid-activated (it is expected to be raw logits).
    2. Both `y_hat` and `y` are center-cropped to `panel_shape`. If the
       inputs arrive larger than `panel_shape` (e.g. the dataloader pads
       panels up to a multiple of the patch size/stride for sliding-window
       inference), this removes that padding, so `panel_shape` must be the
       true *unpadded* panel shape.
    3. Rows are cropped to `[upper_bound:lower_bound]` in that now-unpadded
       coordinate frame (the paper only evaluates this row range of each
       Thebe panel).
    4. Both are downsampled by `scale_factor` via bilinear interpolation.
    5. `y` is binarized at 0.5.

    Parameters
    ----------
    y_hat : torch.Tensor
        Raw (pre-sigmoid) model output, shape (B, H, W) or (B, 1, H, W).
    y : torch.Tensor
        Ground-truth fault mask, same spatial shape as `y_hat`.
    panel_shape : Tuple[int, int]
        True, unpadded panel shape `y_hat`/`y` are center-cropped to before
        the row crop (removes any padding added upstream, e.g. by the
        dataloader, for patch tiling).
    upper_bound, lower_bound : int
        Row range kept after cropping to `panel_shape`.
    scale_factor : float
        Downsampling factor applied after the row crop.
    n_thresholds : int
        Number of probability thresholds swept for the PR/F1 curve.
    max_dist : float
        Boundary matching tolerance, as a fraction of the (cropped,
        downsampled) image diagonal.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], BSDSBatchState]
        The cropped/resized prediction probabilities and binary ground
        truth (so the caller can still compute a loss on them); a metrics
        dict with keys `bsds_ods`, `bsds_ods_threshold`,
        `bsds_precision_at_ods`, `bsds_recall_at_ods`, `bsds_ois`,
        `bsds_precision_at_ois`, `bsds_recall_at_ois`, `bsds_auprc`
        (this batch only, see "Scope and caveats" above); and this batch's
        `BSDSBatchState`, for accumulating the true epoch-level metrics.
    """
    if y_hat.ndim == 3:
        y_hat = y_hat.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)

    y_hat = torch.sigmoid(y_hat.float()).clamp(0.0, 1.0)
    y = y.float()

    def _center_crop_to_shape(t: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        target_h, target_w = shape
        h, w = t.shape[-2:]

        if (h, w) == (target_h, target_w):
            return t

        if h < target_h or w < target_w:
            raise ValueError(
                f"Cannot crop tensor with shape {(h, w)} to larger shape {(target_h, target_w)}."
            )

        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return t[..., top : top + target_h, left : left + target_w]

    # Remove any upstream (e.g. dataloader) padding by center-cropping down
    # to the true unpadded panel shape, then keep only the evaluated row
    # range in that unpadded coordinate frame, then downsample.
    y_hat = _center_crop_to_shape(y_hat, panel_shape)
    y = _center_crop_to_shape(y, panel_shape)

    y_hat = y_hat[..., upper_bound:lower_bound, :]
    y = y[..., upper_bound:lower_bound, :]

    ret_y_hat = y_hat
    ret_y = y

    if scale_factor != 1.0:
        out_h = max(1, int(y_hat.shape[-2] / scale_factor))
        out_w = max(1, int(y_hat.shape[-1] / scale_factor))

        y_hat = F.interpolate(
            y_hat,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )

        y = F.interpolate(
            y,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )

    # Bilinear resize can overshoot [0, 1] slightly at edges.
    y_hat = torch.clamp(y_hat, 0.0, 1.0)
    y = y > 0.5

    # Boundary matching tolerance, following BSDS-style relative max_dist.
    h, w = y.shape[-2:]
    max_dist_px = max(1, int(round(max_dist * float((h**2 + w**2) ** 0.5))))
    kernel_size = 2 * max_dist_px + 1

    def _dilate(mask: torch.Tensor) -> torch.Tensor:
        return (
            F.max_pool2d(
                mask.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=max_dist_px,
            )
            > 0
        )

    thresholds = torch.linspace(
        1.0 / float(n_thresholds + 1),
        float(n_thresholds) / float(n_thresholds + 1),
        int(n_thresholds),
        device=y_hat.device,
    )

    eps = 1e-8
    gt = y
    gt_dilated = _dilate(gt)
    batch_size = y_hat.shape[0]
    gt_count = gt.flatten(1).sum(dim=1).float()  # (B,), constant across thresholds

    # Per-sample, per-threshold counts. Keeping the batch dimension lets ODS
    # (pool over samples, then threshold) and OIS (per-sample best
    # threshold, then pool) use their correct, distinct reductions instead
    # of collapsing to the same statistic.
    count_r = torch.zeros((batch_size, n_thresholds), device=y_hat.device)
    sum_r = torch.zeros((batch_size, n_thresholds), device=y_hat.device)
    count_p = torch.zeros((batch_size, n_thresholds), device=y_hat.device)
    sum_p = torch.zeros((batch_size, n_thresholds), device=y_hat.device)

    for t_idx, threshold in enumerate(thresholds):
        pred = y_hat >= threshold
        pred_dilated = _dilate(pred)

        # A predicted pixel is a hit if it falls close to a GT pixel.
        matched_pred = pred & gt_dilated
        # A GT pixel is recovered if some predicted pixel falls close to it.
        matched_gt = gt & pred_dilated

        count_p[:, t_idx] = matched_pred.flatten(1).sum(dim=1).float()
        sum_p[:, t_idx] = pred.flatten(1).sum(dim=1).float()
        count_r[:, t_idx] = matched_gt.flatten(1).sum(dim=1).float()
        sum_r[:, t_idx] = gt_count

    # ---- ODS: pool counts over the batch, then pick one global threshold.
    count_r_pooled = count_r.sum(dim=0)
    sum_r_pooled = sum_r.sum(dim=0)
    count_p_pooled = count_p.sum(dim=0)
    sum_p_pooled = sum_p.sum(dim=0)

    precision_curve = count_p_pooled / (sum_p_pooled + eps)
    recall_curve = count_r_pooled / (sum_r_pooled + eps)
    f1_curve = (
        2.0 * precision_curve * recall_curve / (precision_curve + recall_curve + eps)
    )

    ods_idx = int(torch.argmax(f1_curve))
    ods = f1_curve[ods_idx]
    ods_threshold = thresholds[ods_idx]
    precision_at_ods = precision_curve[ods_idx]
    recall_at_ods = recall_curve[ods_idx]

    # ---- OIS: each sample picks its own best threshold, then the counts at
    # those (possibly different) thresholds are pooled before computing one
    # F1 - the paper's definition, not a mean of per-sample F1 scores.
    precision_per_sample = count_p / (sum_p + eps)
    recall_per_sample = count_r / (sum_r + eps)
    f1_per_sample = (
        2.0
        * precision_per_sample
        * recall_per_sample
        / (precision_per_sample + recall_per_sample + eps)
    )
    best_idx = f1_per_sample.argmax(dim=1)
    sample_idx = torch.arange(batch_size, device=y_hat.device)

    count_r_best = count_r[sample_idx, best_idx].sum()
    sum_r_best = sum_r[sample_idx, best_idx].sum()
    count_p_best = count_p[sample_idx, best_idx].sum()
    sum_p_best = sum_p[sample_idx, best_idx].sum()

    precision_ois = count_p_best / (sum_p_best + eps)
    recall_ois = count_r_best / (sum_r_best + eps)
    ois = 2.0 * precision_ois * recall_ois / (precision_ois + recall_ois + eps)

    # ---- AUPRC: trapezoidal area under the pooled (ODS) precision-recall curve.
    # follows https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/classification/average_precision.py#L72
    precision_curve = torch.cat([precision_curve, torch.ones(1, dtype=precision_curve.dtype, device=precision_curve.device)])
    recall_curve = torch.cat([recall_curve, torch.zeros(1, dtype=recall_curve.dtype, device=recall_curve.device)])
    auprc = -torch.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[:-1])

    metrics = {
        "bsds_ods": ods,
        "bsds_ods_threshold": ods_threshold,
        "bsds_precision_at_ods": precision_at_ods,
        "bsds_recall_at_ods": recall_at_ods,
        "bsds_ois": ois,
        "bsds_precision_at_ois": precision_ois,
        "bsds_recall_at_ois": recall_ois,
        "bsds_auprc": auprc,
    }

    batch_state = BSDSBatchState(
        thresholds=thresholds,
        count_r=count_r_pooled,
        sum_r=sum_r_pooled,
        count_p=count_p_pooled,
        sum_p=sum_p_pooled,
        count_r_best=count_r_best,
        sum_r_best=sum_r_best,
        count_p_best=count_p_best,
        sum_p_best=sum_p_best,
    )

    return ret_y_hat.float(), ret_y.long(), metrics, batch_state


class BSDSInference(DeepRockModel):
    """DeepRockModel that additionally scores test predictions with the
    BSDS boundary-matching benchmark (ODS/OIS/AUPRC).

    BSDS only makes sense at test time (it evaluates full reconstructed
    panels, not training patches), so this subclass changes nothing about
    training/validation. It overrides `_single_step` to call
    `bsds_inference` when `step_name == "test"`, logging the resulting
    per-batch metrics the same way `DeepRockModel` logs every other metric.
    See `bsds_inference` for what BSDS measures, how it's computed here, and
    its caveats relative to the paper's reference evaluation in
    `deeprock/utils/bsds2d.py`.

    Per-batch BSDS numbers are not the same as the paper's dataset-level
    ODS/OIS (see `bsds_inference`'s docstring), so this class also
    accumulates each test batch's `BSDSBatchState` and, in
    `on_test_epoch_end`, reduces the accumulated counts to true epoch-level
    `epoch_test_bsds_*` metrics before delegating to
    `DeepRockModel.on_test_epoch_end` for the regular torchmetrics-based
    epoch metrics. This mirrors `DeepRockModel._epoch_metrics`'s own
    pattern: accumulate state silently across steps, reduce once, log once,
    then reset for the next epoch.
    """

    def __init__(
        self,
        *args: Any,
        panel_shape: Tuple[int, int] = THEBE_ORIGINAL_PANEL_SHAPE,
        upper_bound: int = THEBE_BSDS_UPPER_BOUND,
        lower_bound: int = THEBE_BSDS_LOWER_BOUND,
        scale_factor: float = THEBE_BSDS_SCALEFACTOR,
        n_thresholds: int = THEBE_BSDS_N_THRESHOLDS,
        max_dist: float = THEBE_BSDS_MAX_DIST,
        bsds_path: PathLike = None,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        *args, **kwargs
            Forwarded to `DeepRockModel.__init__`.
        panel_shape : Tuple[int, int], default=THEBE_ORIGINAL_PANEL_SHAPE
            True, unpadded panel shape passed to `bsds_inference`; used to
            crop away any padding added upstream (e.g. by the dataloader
            for patch tiling) before applying `upper_bound`/`lower_bound`.
        upper_bound, lower_bound : int
            Row range (in unpadded panel coordinates) evaluated by BSDS.
        scale_factor : float
            Downsampling factor applied after the row crop.
        n_thresholds : int
            Number of probability thresholds swept for the PR/F1 curve.
        max_dist : float
            Boundary matching tolerance, as a fraction of the (cropped,
            downsampled) image diagonal.
        """
        super().__init__(*args, **kwargs)
        self.panel_shape = panel_shape
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.scale_factor = scale_factor
        self.n_thresholds = n_thresholds
        self.max_dist = max_dist
        # Running totals of BSDSBatchState across the current test epoch;
        # lazily created on the first test batch, reset once consumed.
        self._bsds_epoch_state: Optional[Dict[str, torch.Tensor]] = None
        self.bsds_path = bsds_path
        self.cached_results = {}

    def on_test_epoch_start(self):
        self._bsds_epoch_state = None

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        x, y, metadata = batch
        if x.ndim == 5:  # patch inference
            x = x.squeeze(0)
        y_hat = self.forward(x)

        if "points" in metadata:  # Patch Inference
            y_hat = (
                self._remove_padding(y_hat, metadata["pad"][0])
                if "pad" in metadata
                else y_hat
            )
            y = (
                self._remove_padding(y, metadata["pad_label"][0])
                if "pad_label" in metadata
                else y
            )
            y_hat_full = torch.zeros(
                y_hat.shape[1], *y.shape[-2:], dtype=y_hat.dtype, device=y_hat.device
            )
            weights = torch.zeros_like(y_hat_full)
            if step_name != "test":
                # At test time we keep raw logits here: bsds_inference
                # applies its own sigmoid after the full panel is
                # reconstructed, so overlapping patches must be averaged as
                # logits, not as probabilities, before it runs.
                y_hat = self._activation_function(y_hat)
            for points, part in zip(metadata["points"][0], y_hat):
                sl = tuple([slice(p, p + s) for p, s in zip([0, *points], part.shape)])
                y_hat_full[sl] += part
                weights[sl] += 1

            y_hat = y_hat_full / weights
            y_hat = (
                y_hat.unsqueeze(0) if y_hat.ndim == 3 else y_hat
            )  # unsqueeze batch_dim

        if self.task_type in ["regression", "binary"]:  # check if it is needed
            y = y.unsqueeze(1) if y.ndim == 3 else y

        bsds_metrics: Dict[str, torch.Tensor] = {}
        if step_name == "test":
            # y_hat/y are complete reconstructed panels (B x C x H x W), not
            # patches, still at the dataloader's padded panel shape.
            # bsds_inference removes that padding, crops/resizes per the
            # Thebe BSDS paper protocol, and returns this batch's
            # boundary-matching metrics (see its docstring for the exact
            # ODS/OIS/AUPRC definitions and their caveats).
            y_hat, y, bsds_metrics, bsds_batch_state = bsds_inference(
                y_hat,
                y,
                panel_shape=self.panel_shape,
                upper_bound=self.upper_bound,
                lower_bound=self.lower_bound,
                scale_factor=self.scale_factor,
                n_thresholds=self.n_thresholds,
                max_dist=self.max_dist,
            )
            self._accumulate_bsds_epoch_state(bsds_batch_state)
            if self.bsds_path is not None:
                self.cached_results[batch_idx] = (y_hat[0, 0].detach().cpu().numpy(), y[0, 0].detach().cpu().numpy())

        loss = self._loss_func(y_hat, y)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if step_name == "train":
            self.running_train_loss(loss)

        metrics = self._compute_metrics(y_hat, y, step_name)
        metrics.update(bsds_metrics)
        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=step_name == "test",
                on_epoch=step_name != "test",
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        return loss

    def _accumulate_bsds_epoch_state(self, state: BSDSBatchState) -> None:
        """Pool one test batch's `BSDSBatchState` into the running epoch
        totals, the BSDS analogue of a torchmetrics `Metric.update()` call.
        """
        if self._bsds_epoch_state is None:
            self._bsds_epoch_state = {
                "thresholds": state.thresholds.detach(),
                "count_r": torch.zeros_like(state.count_r),
                "sum_r": torch.zeros_like(state.sum_r),
                "count_p": torch.zeros_like(state.count_p),
                "sum_p": torch.zeros_like(state.sum_p),
                "count_r_best": torch.zeros_like(state.count_r_best),
                "sum_r_best": torch.zeros_like(state.sum_r_best),
                "count_p_best": torch.zeros_like(state.count_p_best),
                "sum_p_best": torch.zeros_like(state.sum_p_best),
            }

        acc = self._bsds_epoch_state
        acc["count_r"] += state.count_r.detach()
        acc["sum_r"] += state.sum_r.detach()
        acc["count_p"] += state.count_p.detach()
        acc["sum_p"] += state.sum_p.detach()
        acc["count_r_best"] += state.count_r_best.detach()
        acc["sum_r_best"] += state.sum_r_best.detach()
        acc["count_p_best"] += state.count_p_best.detach()
        acc["sum_p_best"] += state.sum_p_best.detach()

    def _sync_bsds_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum a per-rank BSDS accumulator across DDP processes (no-op
        outside DDP), the BSDS analogue of a torchmetrics `Metric.compute()`
        distributed sync."""
        gathered = self.all_gather(tensor)
        if gathered.dim() == tensor.dim() + 1:
            gathered = gathered.sum(dim=0)
        return gathered

    def on_test_epoch_end(self):
        if self._bsds_epoch_state is not None:
            acc = self._bsds_epoch_state
            eps = 1e-8

            count_r = self._sync_bsds_sum(acc["count_r"])
            sum_r = self._sync_bsds_sum(acc["sum_r"])
            count_p = self._sync_bsds_sum(acc["count_p"])
            sum_p = self._sync_bsds_sum(acc["sum_p"])
            count_r_best = self._sync_bsds_sum(acc["count_r_best"])
            sum_r_best = self._sync_bsds_sum(acc["sum_r_best"])
            count_p_best = self._sync_bsds_sum(acc["count_p_best"])
            sum_p_best = self._sync_bsds_sum(acc["sum_p_best"])

            # ODS: one global threshold over pooled counts from every test
            # panel seen this epoch.
            precision_curve = count_p / (sum_p + eps)
            recall_curve = count_r / (sum_r + eps)
            f1_curve = (
                2.0
                * precision_curve
                * recall_curve
                / (precision_curve + recall_curve + eps)
            )
            ods_idx = int(torch.argmax(f1_curve))

            # OIS: pooled counts from every panel's own best threshold.
            precision_ois = count_p_best / (sum_p_best + eps)
            recall_ois = count_r_best / (sum_r_best + eps)
            ois = (
                2.0
                * precision_ois
                * recall_ois
                / (precision_ois + recall_ois + eps)
            )

            # AUPRC over the pooled (ODS) precision-recall curve.
            # follows https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/classification/average_precision.py#L72
            precision_curve = torch.cat([precision_curve, torch.ones(1, dtype=precision_curve.dtype, device=precision_curve.device)])
            recall_curve = torch.cat([recall_curve, torch.zeros(1, dtype=recall_curve.dtype, device=recall_curve.device)])
            auprc = -torch.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[:-1])

            epoch_bsds_metrics = {
                "epoch_test_bsds_ods": f1_curve[ods_idx],
                "epoch_test_bsds_ods_threshold": acc["thresholds"][ods_idx],
                "epoch_test_bsds_precision_at_ods": precision_curve[ods_idx],
                "epoch_test_bsds_recall_at_ods": recall_curve[ods_idx],
                "epoch_test_bsds_ois": ois,
                "epoch_test_bsds_precision_at_ois": precision_ois,
                "epoch_test_bsds_recall_at_ois": recall_ois,
                "epoch_test_bsds_auprc": auprc,
            }
            for metric_name, metric_value in epoch_bsds_metrics.items():
                self.log(
                    metric_name,
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )

            self._bsds_epoch_state = None
        if self.bsds_path is not None:
            try:
                _load_an_evaluate_boundaries(self.bsds_path) # load lib test
                start_time = time.perf_counter()
                ctx = mp.get_context("spawn")
                thresholds = np.linspace(
                    1.0 / float(self.n_thresholds + 1),
                    float(self.n_thresholds) / float(self.n_thresholds + 1),
                    int(self.n_thresholds),
                )

                tasks = [
                    (
                        idx,
                        self.cached_results[idx][0],
                        self.cached_results[idx][1],
                        thresholds,
                        self.bsds_path,
                        self.scale_factor,
                        self.max_dist,
                    )
                    for idx in self.cached_results
                ]
                
                
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running original BSDS may take approximately 15 minutes")
                with ProcessPoolExecutor(
                    max_workers=min(torch.get_num_threads(), len(tasks)),
                    mp_context=ctx,
                ) as executor:
                    results = list(executor.map(bsds_worker, tasks))
                per_sample = {}
                best = {
                    "count_p": 0.0,
                    "sum_p": 0.0,
                    "count_r": 0.0,
                    "sum_r": 0.0,
                }
                overall = {
                    "count_p": np.zeros(self.n_thresholds, dtype=np.float64),
                    "sum_p": np.zeros(self.n_thresholds, dtype=np.float64),
                    "count_r": np.zeros(self.n_thresholds, dtype=np.float64),
                    "sum_r": np.zeros(self.n_thresholds, dtype=np.float64),
                }

                for r in results:

                    overall["count_r"] += r["count_r"]
                    overall["sum_r"] += r["sum_r"]
                    overall["count_p"] += r["count_p"]
                    overall["sum_p"] += r["sum_p"]

                    b = r["best_idx"]

                    best["count_r"] += r["count_r"][b]
                    best["sum_r"] += r["sum_r"][b]
                    best["count_p"] += r["count_p"][b]
                    best["sum_p"] += r["sum_p"][b]

                    per_sample[r["idx"]] = {
                        "ods": float(r["f1"][b]),
                        "ods_threshold": float(thresholds[b]),
                        "precision_at_ods": float(r["prec"][b]),
                        "recall_at_ods": float(r["rec"][b]),
                        "auprc": float(r["auprc"]),
                        "elapsed_seconds": float(r["elapsed"]),
                    }


        
                rec_overall, prec_overall, f1_overall = _compute_rec_prec_f1(
                    overall["count_r"],
                    overall["sum_r"],
                    overall["count_p"],
                    overall["sum_p"],
                )
                ods_idx = int(np.argmax(f1_overall))
                rec_best, prec_best, f1_best = _compute_rec_prec_f1(
                    np.asarray(best["count_r"], dtype=np.float64),
                    np.asarray(best["sum_r"], dtype=np.float64),
                    np.asarray(best["count_p"], dtype=np.float64),
                    np.asarray(best["sum_p"], dtype=np.float64),
                )
                # AUPRC over the pooled (ODS) precision-recall curve.
                # follows https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/classification/average_precision.py#L72
                precision_curve = np.concatenate([prec_overall, [1]])
                recall_curve = np.concatenate([rec_overall, [0]])
                auprc = -np.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[:-1])
                with open(os.path.join(self.logger.log_dir, "bsds_original.json"), "w") as f:
                    json.dump(
                        {
                            "ods": float(f1_overall[ods_idx]),
                            "ods_threshold": float(thresholds[ods_idx]),
                            "precision_at_ods": float(prec_overall[ods_idx]),
                            "recall_at_ods": float(rec_overall[ods_idx]),
                            "ois": float(f1_best),
                            "precision_at_ois": float(prec_best),
                            "recall_at_ois": float(rec_best),
                            "auprc": float(auprc),
                            "n_panels": len(self.cached_results),
                            "elapsed_seconds": float(time.perf_counter() - start_time),
                            "per_sample": dict(sorted(per_sample.items()))
                        },
                        f,
                        indent=4,
                    )
            except Exception as e:
                print(f"Original BSDS eval skipped - {e}")
        super().on_test_epoch_end()


def bsds_worker(args):
    return bsds_inference_original(*args)


def bsds_inference_original(
    idx: int,
    prob: np.ndarray,
    gt: np.ndarray,
    thresholds: np.ndarray,
    eval_boundaries: str,
    scale_factor: float,
    max_dist: float,
) -> dict[str, Any]:

    start_time = time.perf_counter()

    evaluate_boundaries, _ = _load_an_evaluate_boundaries(eval_boundaries)

    if scale_factor != 1.0:
        import cv2
        out_h = max(1, int(prob.shape[0] / scale_factor))
        out_w = max(1, int(prob.shape[1] / scale_factor))

        prob = cv2.resize(
            prob.astype(np.float32, copy=False),
            (out_w, out_h),
        )

        gt = cv2.resize(
            gt.astype(np.float32, copy=False),
            (out_w, out_h),
        )

    prob = np.clip(prob, 0.0, 1.0).astype(np.float32, copy=False)
    gt = (gt > 0.5).astype(np.float32, copy=False)

    count_r, sum_r, count_p, sum_p, _ = evaluate_boundaries(
        prob,
        gt,
        thresholds=thresholds,
        max_dist=float(max_dist),
        apply_thinning=False,
    )

    count_r = np.asarray(count_r, dtype=np.float64)
    sum_r = np.asarray(sum_r, dtype=np.float64)
    count_p = np.asarray(count_p, dtype=np.float64)
    sum_p = np.asarray(sum_p, dtype=np.float64)

    rec, prec, f1 = _compute_rec_prec_f1(
        count_r,
        sum_r,
        count_p,
        sum_p,
    )

    best_idx = int(np.argmax(f1))

    precision_curve = np.concatenate([prec, [1]])
    recall_curve = np.concatenate([rec, [0]])

    auprc = -np.sum(
        (recall_curve[1:] - recall_curve[:-1])
        * precision_curve[:-1]
    )

    return {
        "idx": idx,
        "count_r": count_r,
        "sum_r": sum_r,
        "count_p": count_p,
        "sum_p": sum_p,
        "best_idx": best_idx,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auprc": auprc,
        "elapsed": time.perf_counter() - start_time,
    }


def _compute_rec_prec_f1(
    count_r: np.ndarray,
    sum_r: np.ndarray,
    count_p: np.ndarray,
    sum_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rec = count_r / (sum_r + (sum_r == 0))
    prec = count_p / (sum_p + (sum_p == 0))
    f1 = 2.0 * prec * rec / (prec + rec + ((prec + rec) == 0))
    return rec, prec, f1


def _load_an_evaluate_boundaries(
    pybsds_path: Path | None,
) -> tuple[
    Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    str,
]:
    errors: list[str] = []
    for candidate in _candidate_pybsds_paths(pybsds_path):
        candidate = candidate.resolve()
        if not (candidate / "bsds" / "evaluate_boundaries.py").exists():
            errors.append(f"{candidate}: missing bsds/evaluate_boundaries.py")
            continue
        if not list((candidate / "bsds").glob("correspond_pixels*.so")):
            errors.append(f"{candidate}: compiled correspond_pixels extension not found")
            continue
        try:
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            for module_name in (
                "bsds.evaluate_boundaries",
                "bsds.correspond_pixels",
                "bsds",
            ):
                sys.modules.pop(module_name, None)
            module = importlib.import_module("bsds.evaluate_boundaries")
            return module.evaluate_boundaries, candidate_str
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    detail = "; ".join(errors) if errors else "no candidate paths were checked"
    raise RuntimeError(
        "BSDS evaluation requires the paper's compiled py-bsds500 backend. "
        "Prepare it with `python deeprock/utils/setup_an_pybsds500.py`, "
        "or pass --bsds_pybsds_path pointing to CNNforFaultInterpretation/py-bsds500, "
        "or set AN_PYBSDS500_PATH. "
        f"Load errors: {detail}"
    )


def _candidate_pybsds_paths(pybsds_path: Path | None) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    values: list[Path | str | None] = [
        pybsds_path,
        os.environ.get("AN_PYBSDS500_PATH"),
        repo_root
        / ".cache"
        / "an_pybsds500"
        / "CNNforFaultInterpretation"
        / "py-bsds500",
        repo_root / ".cache" / "an_pybsds500" / "py-bsds500",
        repo_root / "third_party" / "CNNforFaultInterpretation" / "py-bsds500",
        repo_root / "third_party" / "an_pybsds500" / "py-bsds500",
    ]

    candidates: list[Path] = []
    for value in values:
        if value in (None, ""):
            continue
        path = Path(value).expanduser()
        for candidate in (
            path,
            path / "py-bsds500",
            path / "CNNforFaultInterpretation" / "py-bsds500",
        ):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates
