import torch
from torch import nn

class TokenToPatchReconstruction(nn.Module):
    def __init__(
        self,
        token_dim: int = 384,
        patch_size: int = 14,
        output_shape: tuple[int, int] = (1008, 784),
        num_channels: int = 1,
    ):
        """
        Args:
            token_dim: Dimension of each token (embedding size).
            patch_size: Height and width of each patch (assumed square).
            output_shape: Tuple (H_out, W_out) of final reconstructed image.
            num_channels: Number of channels in the reconstructed image.
        """
        super().__init__()
        self.token_dim = token_dim
        self.patch_size = patch_size
        self.output_shape = output_shape
        self.num_channels = num_channels

        H_out, W_out = output_shape
        assert (
            H_out % patch_size == 0 and W_out % patch_size == 0
        ), "Output shape must be divisible by patch size"

        self.grid_size = (
            H_out // patch_size,
            W_out // patch_size,
        )  # (H_patches, W_patches)
        self.num_tokens = self.grid_size[0] * self.grid_size[1]

        # Linear layer mapping each token to a C×H×W patch
        self.token_to_patch = nn.Linear(
            token_dim, num_channels * patch_size * patch_size
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, num_tokens, token_dim]
        Returns:
            Tensor of shape [B, C, H_out, W_out]
        """
        B, N, D = x.shape
        assert N == self.num_tokens, f"Expected {self.num_tokens} tokens but got {N}"
        assert D == self.token_dim, f"Expected token_dim={self.token_dim} but got {D}"

        x = self.token_to_patch(x)  # [B, N, C * pH * pW]

        H_patches, W_patches = self.grid_size
        x = x.view(
            B, H_patches, W_patches, self.num_channels, self.patch_size, self.patch_size
        )
        # [B, H_p, W_p, C, pH, pW]

        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        # [B, C, H_p, pH, W_p, pW]

        x = x.view(
            B,
            self.num_channels,
            H_patches * self.patch_size,
            W_patches * self.patch_size,
        )
        # [B, C, H_out, W_out]

        return x