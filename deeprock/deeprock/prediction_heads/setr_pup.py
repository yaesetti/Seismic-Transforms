from torch import nn


class SETR_PUP(nn.Module):
    """
    SETR Progressive UPsampling Head.

    Input:
        x: [B, embed_dim, H_p, W_p]

    Output:
        [B, out_channels, H, W]
    """

    def __init__(
        self,
        embed_dim,
        out_channels,
        output_shape=None,
        model_resolution=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.output_shape = output_shape
        self.model_resolution = model_resolution

        hidden_dim = embed_dim // 4

        self.block1 = self.conv_block(embed_dim, hidden_dim)
        self.up1 = nn.Upsample(mode="bilinear", align_corners=False)

        self.block2 = self.conv_block(hidden_dim, hidden_dim)
        self.up2 = nn.Upsample(mode="bilinear", align_corners=False)

        self.block3 = self.conv_block(hidden_dim, hidden_dim)
        self.up3 = nn.Upsample(mode="bilinear", align_corners=False)

        self.block4 = self.conv_block(hidden_dim, hidden_dim)
        self.up4 = nn.Upsample(mode="bilinear", align_corners=False)

        self.classifier = nn.Conv2d(
            hidden_dim,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x, dims):
        if self.output_shape is None:
            H, W = dims[-2:]
        else:
            H, W = self.output_shape

        # Progressive output resolutions
        self.up1.size = (H // 8, W // 8)
        self.up2.size = (H // 4, W // 4)
        self.up3.size = (H // 2, W // 2)
        self.up4.size = (H, W)

        x = self.block1(x)
        x = self.up1(x)

        x = self.block2(x)
        x = self.up2(x)

        x = self.block3(x)
        x = self.up3(x)

        x = self.block4(x)
        x = self.up4(x)

        x = self.classifier(x)
        return x

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )