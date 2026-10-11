from minerva.transforms.transform import CastTo, TransformPipeline

from deeprock.datamodules.base import DeepRockDataModule
from deeprock.utils.data import NumpyPatchReader


class FaultsDataModule(DeepRockDataModule):
    task = "faults"
    AVAILABLE_DATASETS = {
        "thebe": {
            "partition": "partition_method_predefined",
            "readers": {
                "train": ((NumpyPatchReader, dict()), (NumpyPatchReader, dict())),
                "test": ((NumpyPatchReader, dict()), (NumpyPatchReader, dict())),
                "val": ((NumpyPatchReader, dict()), (NumpyPatchReader, dict())),
            },
            "data_transforms": TransformPipeline([CastTo("float32")]),
            "label_transforms": TransformPipeline([CastTo("int64")]),
            "out_channels": 1,
        }
    }

    def get_dataset(self, stage):
        if self.dataset == "thebe":
            if stage != "test":
                self.crop_size = None
            else:
                self.crop_size = (96, 96)
        return super().get_dataset(stage)
