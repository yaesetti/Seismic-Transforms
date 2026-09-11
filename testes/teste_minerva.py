# ------------ Python Base ------------
import os
from pathlib import Path

# ------------ Pytorch ------------
import torch
import torch.nn as nn
from torchmetrics import Accuracy, JaccardIndex, F1Score

# ------------ Timm ------------
import timm
import timm.optim
from timm.loss import BinaryCrossEntropy
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------ Lightning ------------
from lightning import Trainer
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

# ------------ Minerva ------------
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone, DeepLabV3, DeepLabV3PredictionHead
from minerva.models.loaders import FromPretrained
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.transforms.transform import TransformPipeline, Transpose, Padding
from minerva.data.readers.numpy_reader import NumpyFolderReader
from minerva.data.data_modules.base import MinervaDataModule
from minerva.data.datasets.base import SimpleDataset

# Experiment Settings
DATASET_ROOT = Path("/petrobr/parceirosbr/spfm/datasets/seismic-datasets/data/tasks/salt_body_segmentation/tgs/processed_data/partition_method_random")



train_dataset = SimpleDataset(
    readers=[
        NumpyFolderReader(
            path=DATASET_ROOT / "train" / "data",
            allow_pickle=True,
        ),
        NumpyFolderReader(
            path=DATASET_ROOT / "train" / "label",
        ),
    ],
    transforms=[
        Repeat(axis=0, n_repetitions=3),  # Transforms to first reader (data)
        None,  # Transforms to second reader (labels)
    ],
)

data_module = MinervaDataModule(
    train_dataset=train_data,
    test_dataset=train_label,
    batch_size=4,
    num_workers=1,
    additional_train_dataloader_kwargs={"drop_last": True},
    name="TGS Dataset",
)
