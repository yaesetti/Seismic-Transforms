from deeprock.tasks.base import RegressionTask
from deeprock.datamodules.seismic_attribute_calculation import (
    SeismicAttributeDataModule,
)


class SeismicAttributeTask(RegressionTask):
    def __init__(self, **datamodule_kwargs):
        super().__init__(SeismicAttributeDataModule, **datamodule_kwargs)
