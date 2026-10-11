from deeprock.tasks.base import RegressionTask
from deeprock.datamodules.super_resolution import SuperResolutionDataModule
from deeprock import loss


class SuperResolutionTask(RegressionTask):
    def __init__(self, **datamodule_kwargs):
        super().__init__(SuperResolutionDataModule, **datamodule_kwargs)

    @property
    def loss(self):
        return loss.SRCombinedLoss()
