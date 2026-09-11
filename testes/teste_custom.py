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

# ------------ Custom ------------
from TGSReader import TGSReader
from TGSDataset import TGSDataModule

# Setting the seed
seed_everything(7)

# Organizing the directories and paths
DATASET_ROOT = Path("/petrobr/parceirosbr/spfm/datasets/seismic-datasets/data/tasks/salt_body_segmentation/tgs/processed_data/partition_method_random")
LOG_DIR = Path("./logs/tgs_finetuning")
CKPT_DIR = Path("./checkpoints/tgs_finetuning")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment settings
NUM_CLASSES = 1
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 4

# Experiment flags
BACKBONE_FREEZE = True
PRED_HEAD_TYPE = 'deeplabv3'

# Pretrain settings
"""
USE_PRETRAINED = True
PRETRAINED_CKPT_PATH = Path()
"""

# Transform pipeline
transform_pipeline = TransformPipeline([
    Padding(128, 128),
    Transpose([2, 0, 1])   # C, H, W format
])

