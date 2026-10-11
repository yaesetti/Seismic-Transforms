import torch


def vit_adapter(
    backbone: torch.nn.Module, input_data: torch.Tensor, task_type: str
) -> torch.Tensor:
    x = backbone.forward_features(input_data)
    if task_type == "classification":
        x = x["x_norm_clstoken"]
    else:  # regression and segmentation use patch tokens and not CLS tokens
        x = x["x_norm_patchtokens"]  # B, H_patches*W_patches, N
        H, W = input_data.shape[-2:]
        H_patches, W_patches = (
            H // backbone.patch_size,
            W // backbone.patch_size,
        )
        x = torch.transpose(x, -2, -1)  # B, N, H_patches*W_patches
        x = x.view(
            *x.shape[:-1], H_patches, W_patches
        ).contiguous()  # B, N, H_patches, W_patches
    return x
