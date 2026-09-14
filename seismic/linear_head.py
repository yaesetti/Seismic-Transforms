from torch import nn

class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init()
        self.linear_head == nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_head(x)
