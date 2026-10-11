from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
from minerva.transforms.transform import CastTo, TransformPipeline


from deeprock.datamodules.base import DeepRockDataModule
from deeprock.utils.data import NumpyFolderReader


class SeismicAttributeDataModule(DeepRockDataModule):
    task = "seismic_attribute_calculation"
    AVAILABLE_DATASETS = {
        "F3": {
            "partition": "partition_method_test_predefined_val_last",
            "readers": {
                "train": (
                    (NumpyFolderReader, dict(sort_method=["text"])),
                    (NumpyFolderReader, dict(sort_method=["text"])),
                ),
                "test": (
                    (NumpyFolderReader, dict(sort_method=["text"])),
                    (NumpyFolderReader, dict(sort_method=["text"])),
                ),
                "val": (
                    (NumpyFolderReader, dict(sort_method=["text"])),
                    (NumpyFolderReader, dict(sort_method=["text"])),
                ),
            },
            "data_transforms": TransformPipeline([CastTo("float32")]),
            "label_transforms": TransformPipeline([CastTo("float32")]),
            "out_channels": 1,
        }
    }

    def __init__(self, attribute: str, **datamodule_kwargs):
        super().__init__(**datamodule_kwargs)
        self.attribute = attribute

    @property
    def label_relpath(self) -> Path:
        return Path("attributes") / self.attribute
