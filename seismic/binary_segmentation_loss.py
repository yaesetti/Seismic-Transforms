import torch.nn as nn

class BinarySegmentationLoss(nn.Module):
    """
    Wraps BCEWithLogitsLoss to protect against frameworks that automatically 
    squeeze the channel dimension out of target masks.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, target):
        # If Minerva squeezed the target to [Batch, H, W], add the channel back -> [Batch, 1, H, W]
        if target.ndim == 3:
            target = target.unsqueeze(1)
            
        # BCE strictly requires targets to be float tensors
        target = target.float()
        
        return self.bce(preds, target)