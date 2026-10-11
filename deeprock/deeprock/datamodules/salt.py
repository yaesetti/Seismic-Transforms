import numpy as np
from minerva.transforms.transform import CastTo, TransformPipeline


from deeprock.datamodules.base import DeepRockDataModule
from deeprock.utils.data import NumpyFolderReader


def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.float32)


class SaltDataModule(DeepRockDataModule):
    task = "salt"
    AVAILABLE_DATASETS = {
        "tgs": {
            "partition": "partition_method_random",
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
            "label_transforms": TransformPipeline([to_binary_mask, CastTo("int64")]),
            "out_channels": 1,
        }
    }
