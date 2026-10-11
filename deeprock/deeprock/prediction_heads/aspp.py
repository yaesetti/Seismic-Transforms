from typing import Optional

from torch import Tensor, nn
from torchvision.models.segmentation.deeplabv3 import ASPP


class DeepLabV3ASPP(nn.Sequential):
    """DeepLabV3 ASPP module."""

    # output_shape and model_resolution are kept for compatibility with other prediction heads.
    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        output_shape: Optional[tuple[int, int]],
        model_resolution: Optional[tuple[int, int]] = None,
        atrous_rates: tuple[int, int, int] = (12, 24, 36),
        interpolate_mode: Optional[str] = "bilinear",
    ):
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.output_shape = output_shape
        self.model_resolution = model_resolution
        self.atrous_rates = atrous_rates
        self.interpolate_mode = interpolate_mode
        super().__init__(
            ASPP(embed_dim, atrous_rates),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, kernel_size=1),
        )

    # dims is unused; it is kept to match the interface of other prediction heads.
    def forward(self, x: Tensor, dims) -> Tensor:
        x = super().forward(x)

        if self.interpolate_mode is None:
            return x

        H, W = dims[-2:] if self.output_shape is None else self.output_shape

        return nn.functional.interpolate(
            x, size=(H, W), mode=self.interpolate_mode, align_corners=False
        )
