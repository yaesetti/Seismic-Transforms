from minerva.transforms.transform import CastTo, TransformPipeline

from deeprock.datamodules.base import DeepRockDataModule
from deeprock.utils.data import NumpyFolderReader


class FaciesDataModule(DeepRockDataModule):
    task = "facies"
    AVAILABLE_DATASETS = {
        "aicrowd_seamai": {
            "partition": "partition_method_test_first",
            "readers": {
                "train": (
                    (
                        NumpyFolderReader,
                        dict(
                            sort_method=["text", "numeric"],
                            delimiter="_",
                            key_index=[0, 1],
                        ),
                    ),
                    (
                        NumpyFolderReader,
                        dict(
                            sort_method=["text", "numeric"],
                            delimiter="_",
                            key_index=[0, 1],
                        ),
                    ),
                ),
                "test": (
                    (
                        NumpyFolderReader,
                        dict(
                            sort_method=["text", "numeric"],
                            delimiter="_",
                            key_index=[0, 1],
                        ),
                    ),
                    (
                        NumpyFolderReader,
                        dict(
                            sort_method=["text", "numeric"],
                            delimiter="_",
                            key_index=[0, 1],
                        ),
                    ),
                ),
                "val": (
                    (
                        NumpyFolderReader,
                        dict(
                            sort_method=["text", "numeric"],
                            delimiter="_",
                            key_index=[0, 1],
                        ),
                    ),
                    (
                        NumpyFolderReader,
                        dict(
                            sort_method=["text", "numeric"],
                            delimiter="_",
                            key_index=[0, 1],
                        ),
                    ),
                ),
            },
            "data_transforms": TransformPipeline([CastTo("float32")]),
            "label_transforms": TransformPipeline([CastTo("int64")]),
            "target_size": (1006, 590),
            "out_channels": 6,
        }
    }
