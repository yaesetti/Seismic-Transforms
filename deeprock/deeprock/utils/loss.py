import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation using logits.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, 1, H, W)
        targets: (B, H, W) or (B, 1, H, W)
        """
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        probs = torch.sigmoid(logits)

        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogits + Dice loss for binary segmentation.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


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
