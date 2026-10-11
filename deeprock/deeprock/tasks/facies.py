from deeprock.tasks.base import SegmentationTask
from deeprock.datamodules.facies import FaciesDataModule


class FaciesTask(SegmentationTask):
    def __init__(self, **datamodule_kwargs):
        super().__init__(FaciesDataModule, **datamodule_kwargs)
