from deeprock.tasks.base import SegmentationTask, BinaryTask
from deeprock.datamodules.channel_identification import (
    BinaryChannelIdentificationDataModule,
    ChannelIdentificationDataModule,
)


class BinaryChannelIdentificationTask(BinaryTask):
    def __init__(self, **datamodule_kwargs):
        super().__init__(BinaryChannelIdentificationDataModule, **datamodule_kwargs)


class ChannelIdentificationTask(SegmentationTask):
    def __init__(self, **datamodule_kwargs):
        super().__init__(ChannelIdentificationDataModule, **datamodule_kwargs)
