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

# ------------ Custom ------------
from seismic.linear_head import LinearSegmentationHead
from seismic.seismic_model import SeismicModel

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Organizing the directories and paths -=-=-=-=-=-=-=-=-=-=-=-=-=-

DATASET_ROOT = Path("/petrobr/parceirosbr/spfm/datasets/seismic-datasets/data/tasks/salt_body_segmentation/tgs/processed_data/partition_method_random")
LOG_DIR = Path("../tgs/logs/tgs_finetuning")
CKPT_DIR = Path("../tgs/checkpoints/tgs_finetuning")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Experiment settings -=-=-=-=-=-=-=-=-=-=-=-=-=-

NUM_CLASSES = 1
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 4

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Experiment flags -=-=-=-=-=-=-=-=-=-=-=-=-=-

BACKBONE_FREEZE_STRATEGY = 'full_freeze' # 'full_freeze' or 'custom_freeze'
PRED_HEAD_TYPE = 'deeplabv3'

USE_META_SSL = True
META_SSL_TYPE = 'dino'

SEED = 7
seed_everything(SEED)

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Datasets -=-=-=-=-=-=-=-=-=-=-=-=-=-

# Transforms that will be applied in the dataset 
transform_pipeline = TransformPipeline([
    Padding(128, 128),
    Transpose([2, 0, 1])
])

# Train Dataset
train_dataset = SimpleDataset(
    readers=[
        NumpyFolderReader(
            path=DATASET_ROOT / "train" / "data",
            allow_pickle=True,
        ),
        NumpyFolderReader(
            path=DATASET_ROOT / "train" / "label",
            allow_pickle=True,
        ),
    ],
    transforms=transform_pipeline,
)

# Val Dataset
val_dataset = SimpleDataset(
    readers=[
        NumpyFolderReader(
            path=DATASET_ROOT / "val" / "data",
            allow_pickle=True,
        ),
        NumpyFolderReader(
            path=DATASET_ROOT / "val" / "label",
            allow_pickle=True,
        ),
    ],
    transforms=transform_pipeline,
)

# Test Dataset
test_dataset = SimpleDataset(
    readers=[
        NumpyFolderReader(
            path=DATASET_ROOT / "test" / "data",
            allow_pickle=True,
        ),
        NumpyFolderReader(
            path=DATASET_ROOT / "test" / "label",
            allow_pickle=True,
        ),
    ],
    transforms=transform_pipeline,
)

# Data Module
data_module = MinervaDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=4,
    num_workers=1,
    additional_train_dataloader_kwargs={"drop_last": True},
    additional_val_dataloader_kwargs={"drop_last": True},
    additional_test_dataloader_kwargs={"drop_last": True},
    name="TGS Dataset",
)

print(data_module)

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Backbone and Weights -=-=-=-=-=-=-=-=-=-=-=-=-=-

deeplab_backbone = DeepLabV3Backbone(num_classes=NUM_CLASSES)

if USE_META_SSL:
    print(f"Downloading/Loading Meta's {META_SSL_TYPE.upper()} SSL weights for ResNet50...")

    if META_SSL_TYPE == 'dino':
        # Load Meta's DINO ResNet50
        meta_resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif META_SSL_TYPE == 'swav':
        # Load Meta's SwAV ResNet50
        meta_resnet = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    else:
        raise ValueError("Invalid SSL type. Choose 'dino' or 'swav'")
    
    # Extract the state dictionary from the downloaded Meta model
    meta_state_dict = meta_resnet.state_dict()

    # Load the weights into your DeepLabV3 Backbone
    # strict=False is REQUIRED because DeepLabV3 modifies the standard ResNet50 architecture 
    # (it changes strides and adds dilations in layer3 and layer4). 
    # However, the convolution weight matrices are the exact same shape, so they load safely.
    incompatible_keys = deeplab_backbone.load_state_dict(meta_state_dict, strict=False)
    
    print("\n--- Meta SSL Transfer Learning Check ---")
    print(f"Missing keys (expected for DeepLab/classification heads): {len(incompatible_keys.missing_keys)}")
    print(f"Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
    
    # Optional: Log the exact keys if you need to debug your research
    # print("Missing:", incompatible_keys.missing_keys)

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Prediction Head -=-=-=-=-=-=-=-=-=-=-=-=-=-

if PRED_HEAD_TYPE == 'deeplabv3':
    pred_head = DeepLabV3PredictionHead(num_classes=NUM_CLASSES)
else:
    pred_head = LinearSegmentationHead(in_channels=2, num_classes=NUM_CLASSES)

val_metrics = {
    "IoU": JaccardIndex(task='binary'),
    "acc": Accuracy(task='binary'),
    "f1-weighted": F1Score(task='binary', average='weighted')
}

# -=-=-=-=-=-=-=-=-=-=-=-=-=- Model and Parameters -=-=-=-=-=-=-=-=-=-=-=-=-=-

training_parameters = {
    'backbone': deeplab_backbone,
    'pred_head': pred_head,
    'num_classes': NUM_CLASSES,
    'val_metrics': val_metrics,
    'optimizer': torch.optim.AdamW,
    'optimizer_kwargs': {
        'weight_decay': 1e-4,
        'lr': LEARNING_RATE,
    },
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
    monitor='val_mIoU',
    mode='max',
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

metrics = {
    "IoU": JaccardIndex(task='binary'),
    "acc": Accuracy(task='binary'),
    "f1-weighted": F1Score(task='binary', average='weighted')
}

eval_pipeline = SimpleLightningPipeline(
    model=model,
    trainer=trainer,
    log_dir=LOG_DIR,
    save_run_status=True,
    seed=SEED,
    apply_metrics_per_sample=False,
    classification_metrics=metrics
)

eval_pipeline.run(data_module, task='evaluate', ckpt_path=CKPT_DIR)
