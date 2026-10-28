import copy
from datetime import datetime
from contextlib import contextmanager
import os
from pathlib import Path
from typing import List, Union, Literal, Callable, Optional, Tuple, Dict, Any

import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from minerva.utils.typing import PathLike

from deeprock.tasks import (
    DeepRockTask,
    BinaryChannelIdentificationTask,
    ChannelIdentificationTask,
    FaciesTask,
    FaultsTask,
    SaltTask,
    SeismicAttributeTask,
    SuperResolutionTask,
)
from deeprock.model import DeepRockModel
from deeprock.model_bsds import BSDSInference
from deeprock.prediction_heads.linear import LinearSegmentationHead
from deeprock.prediction_heads.mlp import MLPSegmentationHead
from deeprock.prediction_heads.aspp import DeepLabV3ASPP
from deeprock.prediction_heads.setr_pup import SETR_PUP

TASKS: Dict[str, Tuple[Callable[..., DeepRockTask], Dict[str, Any]]] = {
    "binary_channel_identification:cigChannel": (
        BinaryChannelIdentificationTask,
        {"dataset": "cigChannel", "batch_size": 128, "eval_batch_size": 256},
    ),
    "channel_identification:cigChannel": (
        ChannelIdentificationTask,
        {"dataset": "cigChannel", "batch_size": 128, "eval_batch_size": 256},
    ),
    "facies:aicrowd_seamai": (FaciesTask, {"dataset": "aicrowd_seamai"}),
    "faults:thebe": (FaultsTask, {"dataset": "thebe", "batch_size": 1024}),
    "salt:tgs": (SaltTask, {"dataset": "tgs"}),
    "seismic_attribute_computation:F3:envelope": (
        SeismicAttributeTask,
        {"dataset": "F3", "attribute": "envelope"},
    ),
    "seismic_attribute_computation:F3:semblance": (
        SeismicAttributeTask,
        {"dataset": "F3", "attribute": "semblance"},
    ),
    "super_resolution:seismic_super_resolution": (
        SuperResolutionTask,
        {"dataset": "seismic_super_resolution"},
    ),
}

PREDICTION_HEADS: Dict[str, Callable[..., nn.Module]] = {
    "linear": LinearSegmentationHead,
    "mlp": MLPSegmentationHead,
    "aspp": DeepLabV3ASPP,
    "pup": SETR_PUP,
}


@contextmanager
def mark_done(done_file: str):
    try:
        yield
    except Exception:
        raise
    else:
        done_file.parent.mkdir(parents=True, exist_ok=True)
        with open(done_file, "w") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(now)


def checkpointing_callbacks(ckpt_path: Path):
    checkpoint_val = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="best-val-{epoch:02d}-{step:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,  # Only save weights to reduce storage space
        enable_version_counter=False,  # Disable version counter to always overwrite the best val checkpoint
    )
    checkpoint_train = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="best-train-{epoch:02d}-{step:02d}-{train_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,  # Only save weights to reduce storage space
        enable_version_counter=False,  # Disable version counter to always overwrite the best train checkpoint
    )
    checkpoint_last = ModelCheckpoint(
        dirpath=ckpt_path,
        save_last=True,
        save_top_k=0,
        monitor=None,
        save_weights_only=False,  # Save full checkpoint for the last one to allow resuming training if needed
        enable_version_counter=False,  # Disable version counter to always overwrite the last checkpoint
    )
    return checkpoint_val, checkpoint_train, checkpoint_last


def locate_checkpoints(ckpt_dir: Path):
    best_val_ckpt = None
    best_val_timestamp = 0
    best_train_ckpt = None
    best_train_timestamp = 0
    last_ckpt = None
    last_timestamp = 0
    

    for ckpt in ckpt_dir.glob("*.ckpt"):
        ckpt_timestamp = os.path.getmtime(ckpt)
        if "best-val" in ckpt.name:
            if ckpt_timestamp > best_val_timestamp:
                best_val_ckpt = ckpt
                best_val_timestamp = ckpt_timestamp
        elif "best-train" in ckpt.name:
            if ckpt_timestamp > best_train_timestamp:
                best_train_ckpt = ckpt
                best_train_timestamp = ckpt_timestamp
        elif "last" in ckpt.name:
            if ckpt_timestamp > last_timestamp:
                last_ckpt = ckpt
                last_timestamp = ckpt_timestamp

    return best_val_ckpt, best_train_ckpt, last_ckpt


def benchmark(
    backbone: nn.Module,
    task: str,
    data_dir: PathLike,
    logs_dir: PathLike,
    prediction_head: Union[
        Literal["linear", "mlp", "aspp", "pup"], nn.Module
    ] = "linear",
    backbone_adapter: Optional[Callable] = None,
    epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    learning_rate: Union[float, Tuple[float, float]] = 1e-4,
    freeze_backbone: bool = False,
    optimizer: type = torch.optim.Adam,
    norm: Optional[Literal["znorm", "clipnorm"]] = None,
    domain_norm: Optional[List[Tuple[int, int]]] = None,
    model_resolution: Optional[int] = None,
    crop_size: Optional[Tuple[int, int]] = None,
    channels: int = 3,
    enforce_interpolation: bool = True,
    accelerator: Literal["cpu", "gpu"] = "gpu",
    devices: Union[str, int, List[int]] = "auto",
    precision: str = "32-true",
    execution_id: int = 0,
    stages: Literal["train", "evaluate", "train+evaluate"] = "train+evaluate",
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    experiment_metadata: Optional[str] = None,
    debug_mode: bool = False,
    subset_type: Optional[Literal["random", "binary_tree", "first_n"]] = None,
    subset_size: Optional[int] = None,
    subset_kwargs: Optional[Dict[str, Any]] = None,
    drop_last_train: bool = True,
    progress_bar_refresh_rate: Optional[int] = 1,
    skip_completed: bool = False,
    dry_run: bool = False,
    bsds_path: PathLike = None,
):
    if max_steps is None and epochs is None:
        raise ValueError("Either num_steps or epochs must be provided!")

    # Check if num_steps and epochs are mutually exclusive
    if max_steps is not None and epochs is not None:
        raise ValueError(
            "num_steps and epochs are mutually exclusive! Please provide only one of them."
        )
    if norm not in ["clipnorm", "znorm", None]:
        raise ValueError(
            "Invalid normalization. 'norm' can only be one of ['clipnorm', 'znorm', None]"
        )

    # Cast to Path objects
    data_dir = Path(data_dir)
    logs_dir = Path(logs_dir)

    base_task = task.split('+')[0]

    selected_task = TASKS.get(base_task)
    if selected_task is None:
        raise ValueError(
            f"Invalid task: {task}! Task must be one of the following: {', '.join(TASKS)}"
        )

    # Task kwargs is the dataset kwargs for the selected
    task_kwargs = selected_task[1]
    # override batch size if provided as argument
    if batch_size is not None:
        task_kwargs["batch_size"] = batch_size
    # override eval batch size if provided as argument
    if eval_batch_size is not None:
        task_kwargs["eval_batch_size"] = eval_batch_size

    if subset_type is not None:
        task_kwargs["train_subset_type"] = subset_type
        task_kwargs["train_subset_size"] = subset_size
        task_kwargs["train_subset_kwargs"] = subset_kwargs or {}
    # This is useful when subset samples is less than batch size, and we want to avoid dropping the last batch
    task_kwargs["drop_last_train"] = drop_last_train

    deeprock_task = selected_task[0](
        data_dir=data_dir,
        norm=norm,
        domain_norm=domain_norm,
        model_resolution=model_resolution,
        crop_size=crop_size,
        channels=channels,
        **task_kwargs,
    )

    deeprock_task.setup()

    if deeprock_task.task_type is None:
        raise ValueError("Task type must be defined in the task class!")

    meta_data = (
        torch.from_numpy(deeprock_task.datamodule.train_dataset[0][0])  # type: ignore
        .unsqueeze(0)
        .to("meta")
    )
    meta_backbone = copy.deepcopy(backbone).to("meta")
    meta_output = (
        meta_backbone(meta_data)
        if backbone_adapter is None
        else backbone_adapter(meta_backbone, meta_data, deeprock_task.task_type)
    )
    if isinstance(prediction_head, nn.Module):
        meta_ph = copy.deepcopy(prediction_head).to("meta")
        try:
            meta_ph(meta_output)
        except:
            raise RuntimeError("Prediction Head is incompatible with backbone output!")
    else:
        embed_dim = meta_output.shape[1]
        prediction_head = PREDICTION_HEADS[prediction_head](
            embed_dim=embed_dim,
            out_channels=deeprock_task.out_channels,
            output_shape=deeprock_task.output_shape,
            model_resolution=None if enforce_interpolation else model_resolution,
        )

    if isinstance(learning_rate, float):
        learning_rate = [learning_rate, learning_rate]  # type: ignore
    elif len(learning_rate) == 1:  # type: ignore
        learning_rate = [learning_rate[0], learning_rate[0]]  # type: ignore
    elif len(learning_rate) > 2:  # type: ignore
        raise ValueError(
            "At most 2 learning rate values can be provided! One for the backbone and the second one for the prediction_head"
        )

    if freeze_backbone:
        learning_rate[0] = 0  # type: ignore

    if isinstance(learning_rate, list):
        learning_rate = tuple(learning_rate)  # type: ignore

    model_cls = BSDSInference if base_task == "faults:thebe" else DeepRockModel
    model = model_cls(
        backbone=backbone,
        prediction_head=prediction_head,
        adapter=backbone_adapter,
        loss_fn=deeprock_task.loss,
        task_type=deeprock_task.task_type,
        learning_rate=learning_rate,  # type: ignore
        train_metrics=deeprock_task.train_metrics,
        val_metrics=deeprock_task.val_metrics,
        test_metrics=deeprock_task.test_metrics,
        optimizer=optimizer,
        bsds_path=bsds_path,
    )

    # Dry run only checks model instantiation and compatibility with input
    if dry_run:
        return

    logs_dir = Path(logs_dir) / os.path.join(*task.lower().split(":"))
    ckpt_dir = logs_dir / str(execution_id) / "checkpoints"

    if experiment_metadata is not None:
        config_dir = logs_dir / str(execution_id) / "config.yaml"
        config_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(config_dir, "w") as f:
            f.write(experiment_metadata)
        print(f"Experiment metadata saved to {config_dir}")
    # Set seed as same as the execution_id
    L.seed_everything(execution_id, workers=True)
    train_done_file = logs_dir / str(execution_id) / "TRAIN_DONE"
    evaluate_done_file = logs_dir / str(execution_id) / "EVALUATE_DONE"

    if stages in ["train", "train+evaluate"]:
        if train_done_file.exists() and skip_completed:
            print(
                f"Training already completed for execution_id {execution_id}. Skipping training..."
            )
        else:
            if train_done_file.exists():
                print(
                    f"Training already completed for execution_id {execution_id}. Re-running training..."
                )
                train_done_file.unlink(
                    missing_ok=True
                )  # Remove the TRAIN_DONE file to allow re-running training
            # As train changes, we need to remove the EVALUATE_DONE file to allow re-running evaluation
            evaluate_done_file.unlink(
                missing_ok=True
            )  # Remove the EVALUATE_DONE file to allow re-running evaluation

            with mark_done(train_done_file):
                train(
                    logs_dir=logs_dir,
                    epochs=epochs,
                    max_steps=max_steps,
                    accelerator=accelerator,
                    devices=devices,
                    precision=precision,
                    execution_id=execution_id,
                    debug_mode=debug_mode,
                    progress_bar_refresh_rate=progress_bar_refresh_rate,
                    deeprock_task=deeprock_task,
                    model=model,
                    ckpt_dir=ckpt_dir,
                )

    if stages in ["evaluate", "train+evaluate"]:
        if evaluate_done_file.exists() and skip_completed:
            print(
                f"Evaluation already completed for execution_id {execution_id}. Skipping evaluation..."
            )
        else:
            if evaluate_done_file.exists():
                print(
                    f"Evaluation already completed for execution_id {execution_id}. Re-running evaluation..."
                )
                evaluate_done_file.unlink(
                    missing_ok=True
                )  # Remove the EVALUATE_DONE file to allow re-running evaluation

            # Evaluate Checkpoints
            with mark_done(evaluate_done_file):
                evaluate(
                    logs_dir=logs_dir,
                    accelerator=accelerator,
                    devices=devices,
                    precision=precision,
                    execution_id=execution_id,
                    deeprock_task=deeprock_task,
                    model=model,
                    ckpt_dir=ckpt_dir,
                )


def train(
    logs_dir,
    epochs,
    max_steps,
    accelerator,
    devices,
    precision,
    execution_id,
    debug_mode,
    progress_bar_refresh_rate,
    deeprock_task,
    model,
    ckpt_dir,
):

    CSVlogger = CSVLogger(save_dir=logs_dir, name=str(execution_id), version="train")
    checkpoint_val_cb, checkpoint_train_cb, checkpoint_last_cb = (
        checkpointing_callbacks(ckpt_dir)
    )

    if debug_mode:
        print(
            "Debug mode is ON. Reducing the number of batches for training and validation to 10."
        )
        additional_trainer_args = {
            "limit_train_batches": 10,
            "limit_val_batches": 10,
            "limit_test_batches": 10,
            "limit_predict_batches": 10,
        }
    else:
        additional_trainer_args = {}

    if max_steps is not None:
        additional_trainer_args["max_steps"] = max_steps
        additional_trainer_args["val_check_interval"] = max_steps // 10
        additional_trainer_args["check_val_every_n_epoch"] = None
    else:
        additional_trainer_args["max_epochs"] = epochs

    callbacks = [checkpoint_val_cb, checkpoint_train_cb, checkpoint_last_cb]
    if progress_bar_refresh_rate is not None:
        tqdm_progress_bar = TQDMProgressBar(refresh_rate=progress_bar_refresh_rate)
        callbacks.append(tqdm_progress_bar)
    else:
        print(
            "Progress bar refresh rate is set to None. Progress bar will not be displayed."
        )
        additional_trainer_args["enable_progress_bar"] = False

    trainer = L.Trainer(
        logger=CSVlogger,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        precision=precision,  # type: ignore
        **additional_trainer_args,
    )

    _, _, last_ckpt = locate_checkpoints(ckpt_dir)

    # Train + Val
    trainer.fit(model, deeprock_task.datamodule, ckpt_path=last_ckpt)


def evaluate(
    logs_dir,
    accelerator,
    devices,
    precision,
    execution_id,
    deeprock_task,
    model,
    ckpt_dir,
):
    checkpoint_val, checkpoint_train, checkpoint_last = locate_checkpoints(ckpt_dir)
    if checkpoint_last is not None:
        deeprock_task.eval(
            model=model,
            logs_dir=logs_dir,
            execution_id=execution_id,
            ckpt_path=checkpoint_last,
            ckpt_name="last",
            accelerator=accelerator,
            devices=devices,
            precision=precision,
        )
        # Test Best Val Loss
    if checkpoint_val is not None:
        deeprock_task.eval(
            model=model,
            logs_dir=logs_dir,
            execution_id=execution_id,
            ckpt_path=checkpoint_val,
            ckpt_name="best_val",
            accelerator=accelerator,
            devices=devices,
            precision=precision,
        )
        # Test Best Train Loss
    if checkpoint_train is not None:
        deeprock_task.eval(
            model=model,
            logs_dir=logs_dir,
            execution_id=execution_id,
            ckpt_path=checkpoint_train,
            ckpt_name="best_train",
            accelerator=accelerator,
            devices=devices,
            precision=precision,
        )
