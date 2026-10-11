from minerva.transforms.transform import CastTo, TransformPipeline


from deeprock.datamodules.base import DeepRockDataModule
from deeprock.utils.data import NumpyFolderReader


class SuperResolutionDataModule(DeepRockDataModule):
    task = "super_resolution"
    AVAILABLE_DATASETS = {
        "seismic_super_resolution": {
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
            "label_transforms": TransformPipeline([CastTo("float32")]),
            "out_channels": 1,
            "output_shape": (256, 256),
        }
    }
