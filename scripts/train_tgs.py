# ------------ Python Base ------------
import numpy as np
import sys
import argparse
import re
from pathlib import Path

# ------------ Pytorch ------------
import torch
import torch.nn as nn
from torchmetrics import Accuracy, JaccardIndex, F1Score
from torch.optim.lr_scheduler import CosineAnnealingLR

# ------------ Torchvision ------------
from torchvision.models import resnet50, ResNet50_Weights

# ------------ Timm ------------
import timm
import timm.optim
from timm.loss import BinaryCrossEntropy

# ------------ Lightning ------------
from lightning import Trainer
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

# ------------ Minerva ------------
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone, DeepLabV3, DeepLabV3PredictionHead
from minerva.models.loaders import FromPretrained
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.transforms.transform import TransformPipeline, Transpose, Repeat, CastTo
from minerva.data.readers.numpy_reader import NumpyFolderReader
from minerva.data.data_modules.base import MinervaDataModule
from minerva.data.datasets.base import SimpleDataset

# ------------ Custom ------------
sys.path.append('/petrobr/parceirosbr/home/victor.setti/workspace/Seismic-Transforms')

from seismic.linear_head import LinearSegmentationHead
from seismic.seismic_model import SeismicModel
from seismic.padding import SafePadding
from seismic.transformed_reader import TransformedReader
from seismic.binary_segmentation_loss import BinarySegmentationLoss
from seismic.tgs_metrics import BinaryTGSMeanIoU

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Organizing the directories and paths -=-=-=-=-=-=-=-=-=-=-=-=-=-

DATASET_ROOT = Path("/petrobr/parceirosbr/spfm/datasets/seismic-datasets/data/tasks/salt_body_segmentation/tgs/processed_data/partition_method_random")

parser = argparse.ArgumentParser(description='Train TGS Model')
parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment")
parser.add_argument("--group-name", type=str, default=None, help="Name of the parent grouping directory")

args = parser.parse_args()

EXP_NAME = args.exp_name

if args.group_name:
    OUT_ROOT = Path(f"/petrobr/parceirosbr/home/victor.setti/workspace/Seismic-Transforms/outputs/tgs/{args.group_name}/{EXP_NAME}")
else:
    OUT_ROOT = Path(f"/petrobr/parceirosbr/home/victor.setti/workspace/Seismic-Transforms/outputs/tgs/{EXP_NAME}")

LOG_DIR = OUT_ROOT / "logs"
CKPT_DIR = OUT_ROOT / "checkpoints"
PLOTS_DIR = OUT_ROOT / "plots"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Experiment settings -=-=-=-=-=-=-=-=-=-=-=-=-=-

NUM_CLASSES = 1
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 256
NUM_WORKERS = 20

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Experiment flags -=-=-=-=-=-=-=-=-=-=-=-=-=-

# 'full_freeze' or 'custom_freeze' or 'full_finetuning'
BACKBONE_FREEZE_STRATEGY = 'full_finetuning'

PRED_HEAD_TYPE = 'deeplabv3'

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Seeding -=-=-=-=-=-=-=-=-=-=-=-=-=-

seed_match = re.search(r'\d+', EXP_NAME)
SEED = int(seed_match.group()) if seed_match else 7

print(f"Experiment Name: {EXP_NAME} | Automatically setting SEED to: {SEED}")
seed_everything(SEED)

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Datasets -=-=-=-=-=-=-=-=-=-=-=-=-=-

IGNORE_INDEX = 255

# Transforms that will be applied in the data
data_transform_pipeline = TransformPipeline([
    SafePadding(128, 128, padding_mode='reflect'),
    Transpose([2, 0, 1]),
    Repeat(axis=0, n_repetitions=3)
])

# Transforms that will be applied in the masks
label_transform_pipeline = TransformPipeline([
    SafePadding(
        128,
        128,
        padding_mode='constant',
        padding_value=IGNORE_INDEX,
        mask_padding_value=IGNORE_INDEX
    ),
    Transpose([2, 0, 1]),
    CastTo(np.float32)
])

# Train Dataset
train_dataset = SimpleDataset(
    readers=[
        TransformedReader(
            reader=NumpyFolderReader(
                path=DATASET_ROOT / "train" / "data",
                allow_pickle=True,
            ),
            transform=data_transform_pipeline
        ),
        TransformedReader(
            reader=NumpyFolderReader(
                path=DATASET_ROOT / "train" / "label",
                allow_pickle=True,
            ),
            transform=label_transform_pipeline
        )
    ],
)

# Val Dataset
val_dataset = SimpleDataset(
    readers=[
        TransformedReader(
            reader=NumpyFolderReader(
                path=DATASET_ROOT / "val" / "data",
                allow_pickle=True,
            ),
            transform=data_transform_pipeline
        ),
        TransformedReader(
            reader=NumpyFolderReader(
                path=DATASET_ROOT / "val" / "label",
                allow_pickle=True,
            ),
            transform=label_transform_pipeline
        )
    ],
)

# Test Dataset
test_dataset = SimpleDataset(
    readers=[
        TransformedReader(
            reader=NumpyFolderReader(
                path=DATASET_ROOT / "test" / "data",
                allow_pickle=True,
            ),
            transform=data_transform_pipeline
        ),
        TransformedReader(
            reader=NumpyFolderReader(
                path=DATASET_ROOT / "test" / "label",
                allow_pickle=True,
            ),
            transform=label_transform_pipeline
        )
    ],
)

# Data Module
data_module = MinervaDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    additional_train_dataloader_kwargs={"drop_last": True},
    additional_val_dataloader_kwargs={"drop_last": True},
    additional_test_dataloader_kwargs={"drop_last": False},
    name="TGS Dataset",
)

print(data_module)

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Backbone and Weights -=-=-=-=-=-=-=-=-=-=-=-=-=-

deeplab_backbone = DeepLabV3Backbone(num_classes=NUM_CLASSES)

print("Downloading/Loading TorchVision's ImageNet1K_V2 weights for ResNet50...")

# Load standard ResNet50 with the improved V2 training recipe
tv_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Extract the state dictionary from the downloaded TorchVision model
tv_state_dict = tv_resnet.state_dict()

# Load the weights into your DeepLabV3 Backbone
# strict=False is REQUIRED because DeepLabV3 modifies the standard ResNet50 architecture
incompatible_keys = deeplab_backbone.load_state_dict(tv_state_dict, strict=False)

print("\n--- Imaginet Transfer Learning Check ---")
print(f"Missing keys (expected for DeepLab/classification heads): {len(incompatible_keys.missing_keys)}")
print(f"Unexpected keys: {len(incompatible_keys.unexpected_keys)}")

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Prediction Head -=-=-=-=-=-=-=-=-=-=-=-=-=-

if PRED_HEAD_TYPE == 'deeplabv3':
    pred_head = DeepLabV3PredictionHead(num_classes=NUM_CLASSES)
else:
    pred_head = LinearSegmentationHead(in_channels=2048, num_classes=NUM_CLASSES)


# -=-=-=-=-=-=-=-=-=-=-=-=-=- Metrics -=-=-=-=-=-=-=-=-=-=-=-=-=-

val_metrics = {
    "IoU_Standard": JaccardIndex(task='binary', ignore_index=IGNORE_INDEX),
    "TGS_Benchmark": BinaryTGSMeanIoU(ignore_index=IGNORE_INDEX),
    "acc": Accuracy(task='binary', ignore_index=IGNORE_INDEX),
    "f1-weighted": F1Score(task='binary', average='weighted', ignore_index=IGNORE_INDEX)
}

test_metrics = {
    "IoU_Standard": JaccardIndex(task='binary', ignore_index=IGNORE_INDEX),
    "TGS_Benchmark": BinaryTGSMeanIoU(ignore_index=IGNORE_INDEX),
    "acc": Accuracy(task='binary', ignore_index=IGNORE_INDEX),
    "f1-weighted": F1Score(task='binary', average='weighted', ignore_index=IGNORE_INDEX)
}

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Model and Parameters -=-=-=-=-=-=-=-=-=-=-=-=-=-

training_parameters = {
    'backbone': deeplab_backbone,
    'pred_head': pred_head,
    'num_classes': NUM_CLASSES,
    'val_metrics': val_metrics,
    'test_metrics': test_metrics,
    'loss_fn': BinarySegmentationLoss(ignore_index=IGNORE_INDEX),
    'optimizer': torch.optim.AdamW,
    'optimizer_kwargs': {
        'weight_decay': 1e-4,
        'lr': LEARNING_RATE,
    },
    'lr_scheduler': CosineAnnealingLR,
    'lr_scheduler_kwargs': {
        'T_max': NUM_EPOCHS,
        'eta_min': 1e-6
    }
}

if BACKBONE_FREEZE_STRATEGY == 'full_freeze':
    model = SeismicModel(
        freeze_backbone=True,
        **training_parameters
    )
elif BACKBONE_FREEZE_STRATEGY == 'custom_freeze':
    layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2']
    model = SeismicModel(
        freeze_layers=layers_to_freeze,
        freeze_backbone=False,
        **training_parameters
    )
else:
    model = SeismicModel(
        freeze_backbone=False,
        **training_parameters
    )

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Logging and Checkpoint -=-=-=-=-=-=-=-=-=-=-=-=-=-

csv_logger = CSVLogger(LOG_DIR, name='', version='')

ckpt_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_last=False,
    dirpath=CKPT_DIR,
    filename='best',
    auto_insert_metric_name=False
)

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Trainer -=-=-=-=-=-=-=-=-=-=-=-=-=-

trainer = Trainer(
    logger=csv_logger,
    max_epochs=NUM_EPOCHS,
    limit_val_batches=1.0,
    strategy='auto',
    devices=1,
    check_val_every_n_epoch=1,
    callbacks=[ckpt_callback]
)

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Training -=-=-=-=-=-=-=-=-=-=-=-=-=-

pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=LOG_DIR,
    save_run_status=True
)

pipeline.run(data_module, task='fit')

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Evaluating -=-=-=-=-=-=-=-=-=-=-=-=-=-

eval_pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=LOG_DIR,
    save_run_status=True,
    seed=SEED,
    apply_metrics_per_sample=False,
)

eval_pipeline.run(data_module, task='test', ckpt_path=CKPT_DIR / 'best.ckpt')
