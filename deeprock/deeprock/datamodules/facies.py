from minerva.transforms.transform import CastTo, TransformPipeline
from minerva.transforms.random_transform import RandomFlip, RandomCrop

from deeprock.datamodules.base import DeepRockDataModule
from deeprock.utils.data import NumpyFolderReader


class FaciesDataModule(DeepRockDataModule):
    task = "facies"

    shared_hflip = RandomFlip(num_samples=2, possible_axis=-1, prob=0.5)
    shared_vflip = RandomFlip(num_samples=2, possible_axis=0, prob=0.5)

    shared_crop = RandomCrop(crop_size=(1006, 590), num_samples=2, pad_mode="reflect")

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
        },

        "aicrowd_seamai+hflip": {
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
            "data_transforms": TransformPipeline([
                CastTo("float32"),
                shared_hflip
                ]),
            "label_transforms": TransformPipeline([
                CastTo("int64"),
                shared_hflip
            ]),
            "target_size": (1006, 590),
            "out_channels": 6,
        },

        "aicrowd_seamai+vflip": {
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
            "data_transforms": TransformPipeline([
                CastTo("float32"),
                shared_vflip
                ]),
            "label_transforms": TransformPipeline([
                CastTo("int64"),
                shared_vflip
            ]),
            "target_size": (1006, 590),
            "out_channels": 6,
        }
    }
