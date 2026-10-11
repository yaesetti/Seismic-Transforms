from deeprock.tasks.base import BinaryTask
from deeprock.datamodules.faults import FaultsDataModule


class FaultsTask(BinaryTask):
    def __init__(self, **datamodule_kwargs):
        super().__init__(FaultsDataModule, **datamodule_kwargs)
