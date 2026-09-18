import torch
import torch.nn as nn

class BinarySegmentationLoss(nn.Module):
    """
    Wraps BCEWithLogitsLoss to protect against frameworks that automatically 
    squeeze the channel dimension out of target masks.
    """
    def __init__(self, ignore_index: int | None = None):
        super().__init__()
        self.ignore_index = ignore_index

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, target):
        # If Minerva squeezed the target to [Batch, H, W], add the channel back -> [Batch, 1, H, W]
        if target.ndim == 3:
            target = target.unsqueeze(1)

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
        else:
            valid_mask = torch.ones_like(target, dtype=torch.float32)
            
        # BCE strictly requires targets to be float tensors
        target = target.float()
        
        # 2. Calculate the raw, un-averaged loss for every pixel
        pixel_loss = self.bce(preds, target)
        
        # 3. Multiply by the mask to zero out the loss on padded background pixels
        masked_loss = pixel_loss * valid_mask
        
        # 4. Return the mean loss, strictly averaged ONLY over the valid pixels
        return masked_loss.sum() / valid_mask.sum().clamp_min(1.0)