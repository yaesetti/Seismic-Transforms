# https://github.com/discovery-unicamp/spfm-extra/blob/main/models/linear_head.py
from torch import nn


# TODO move to Minerva
class LinearSegmentationHead(nn.Module):
    def __init__(
        self, embed_dim, out_channels, output_shape=None, model_resolution=None
    ):
        super().__init__()
        self.model_resolution = model_resolution
        self.out_channels = out_channels
        self.output_shape = output_shape
        self.embed_dim = embed_dim

        model_resolution = 1 if model_resolution is None else model_resolution
        self.linear_head = nn.Conv2d(
            embed_dim, out_channels * model_resolution * model_resolution, kernel_size=1
        )

    def forward(self, x, dims):
        # [B, N, H_p, W_p]
        x = self.linear_head(x)
        H, W = dims[-2:] if self.output_shape is None else self.output_shape

        if self.model_resolution is None:
            # [B, C, H_p, W_p]
            x = nn.functional.interpolate(
                x, size=(H, W), mode="bilinear", align_corners=False
            )
            # [B, C, H, W]
        else:
            # [B, C*pH*pW, H_p, W_p]
            B, _, Hp, Wp = x.shape
            x = x.view(
                B,
                self.out_channels,
                self.model_resolution,
                self.model_resolution,
                Hp,
                Wp,
            )
            # [B, C, pH, pW, H_p, W_p]
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            # [B, C, H_p, pH, W_p, pW]
            x = x.view(
                B,
                self.out_channels,
                Hp * self.model_resolution,
                Wp * self.model_resolution,
            )
            # [B, C, H_out, W_out]
        return x
