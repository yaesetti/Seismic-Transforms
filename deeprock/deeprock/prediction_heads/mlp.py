from torch import nn
from torchvision.ops import MLP

class MLPSegmentationHead(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        out_channels, 
        hidden_dim=None, 
        output_shape=None, 
        model_resolution=None,
        dropout=0.0
    ):
        super().__init__()
        self.model_resolution = model_resolution
        self.out_channels = out_channels
        self.output_shape = output_shape
        self.embed_dim = embed_dim

        hidden_dim = hidden_dim or embed_dim
        model_resolution = 1 if model_resolution is None else model_resolution
        
        out_dim = out_channels * model_resolution * model_resolution

        self.mlp_head = MLP(
            in_channels=embed_dim,
            hidden_channels=[hidden_dim, out_dim],
            norm_layer=nn.LayerNorm,
            activation_layer=nn.GELU,
            dropout=dropout
        )

    def forward(self, x, dims):
        # [B, C, H_p, W_p] -> [B, H_p, W_p, C]
        x = x.permute(0, 2, 3, 1)
        
        # [B, H_p, W_p, C] -> [B, H_p, W_p, C_out]
        x = self.mlp_head(x)
        
        # [B, H_p, W_p, C_out] -> [B, C_out, H_p, W_p]
        x = x.permute(0, 3, 1, 2)
        
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
