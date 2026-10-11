from pathlib import Path
import json
import os
from typing import List, Union
from lightning.pytorch.loggers.logger import Logger

import numpy as np
from torchmetrics import (
    # multiclass
    JaccardIndex,
    F1Score,
    # regression
    PeakSignalNoiseRatio,
    MeanSquaredError,
    MeanAbsoluteError,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.classification import (
    # binary
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryMatthewsCorrCoef,
    BinaryAveragePrecision,
)
import lightning as L
from lightning.pytorch.loggers import CSVLogger

from deeprock.metrics import BinaryODS, BinaryOIS, BinaryTGSMeanIoU
from deeprock.model import DeepRockModel
from deeprock import loss


class DeepRockTask:
    task_type = None

    def __init__(self, datamodule_cls, **datamodule_kwargs):
        self.datamodule = None
        self.datamodule_cls = datamodule_cls
        datamodule_kwargs["root_dir"] = datamodule_kwargs.pop("data_dir")
        self.datamodule_kwargs = datamodule_kwargs

    def setup(self):
        self.datamodule = self.datamodule_cls(
            **self.datamodule_kwargs, task_type=self.task_type
        )
        self.datamodule.setup()

    @property
    def train_metrics(self):
        return self.metrics

    @property
    def val_metrics(self):
        return self.metrics

    @property
    def test_metrics(self):
        return self.metrics

    @property
    def metrics(self):
        raise NotImplementedError

    @property
    def loss(self):
        raise NotImplementedError

    @property
    def out_channels(self):
        if self.datamodule is not None:
            return self.datamodule.out_channels
        return None

    @property
    def output_shape(self):
        if self.datamodule is not None:
            return self.datamodule.output_shape
        return None

    def plot(self):
        raise NotImplementedError

    def _interpolate_eval_log_dir(
        self, logs_dir: Path, execution_id: int, ckpt_name: str
    ):
        run_name = f"evaluate/{ckpt_name}"
        return logs_dir / str(execution_id) / "evaluate" / str(ckpt_name)

    def eval(
        self,
        model: DeepRockModel,
        logs_dir: Path,
        execution_id: int,
        ckpt_path: Path,
        ckpt_name: str,
        accelerator: str = "gpu",
        devices: Union[str, int, List[int]] = 1,
        precision: str = "32-true",
        debug_mode: bool = False,
        **kwargs,
    ):
        if debug_mode:
            additional_trainer_args = {
                "limit_train_batches": 10,
                "limit_val_batches": 10,
                "limit_test_batches": 10,
                "limit_predict_batches": 10,
            }
        else:
            additional_trainer_args = {}

        logger = CSVLogger(
            save_dir=logs_dir, name=str(execution_id), version=f"evaluate/{ckpt_name}"
        )
        test_trainer = L.Trainer(
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            **additional_trainer_args,
        )  # type: ignore
        return test_trainer.test(model, datamodule=self.datamodule, ckpt_path=ckpt_path)


class BinaryTask(DeepRockTask):
    task_type = "binary"

    def __init__(self, datamodule_cls, **datamodule_kwargs):
        super().__init__(datamodule_cls, **datamodule_kwargs)
        self.threshold = 0.5

    @property
    def loss(self):
        return loss.BCEDiceLoss(
            bce_weight=0.5,
            dice_weight=0.5,
            pos_weight=None,
        )

    @property
    def metrics(self):
        kwargs = {"threshold": self.threshold, "ignore_index": -1}
        metrics = {
            "pixel_accuracy": BinaryAccuracy(**kwargs),
            "precision": BinaryPrecision(**kwargs),
            "recall": BinaryRecall(**kwargs),
            "specificity": BinarySpecificity(**kwargs),
            "dice_f1": BinaryF1Score(**kwargs),
            "iou_jaccard": BinaryJaccardIndex(**kwargs),
            "mcc": BinaryMatthewsCorrCoef(**kwargs),
            "auprc": BinaryAveragePrecision(
                thresholds=100,
                ignore_index=-1,
            ),
            "tgs-iou": BinaryTGSMeanIoU(pred_threshold=self.threshold, ignore_index=-1),
            "ods": BinaryODS(ignore_index=-1),
            "ois": BinaryOIS(
                ignore_index=-1,
            ),
        }

        return metrics

    def eval(
        self,
        model: DeepRockModel,
        logs_dir: Path,
        execution_id: int,
        ckpt_path: Path,
        ckpt_name: str,
        accelerator: str = "gpu",
        devices: Union[str, int, List[int]] = 1,
        precision: str = "32-true",
        debug_mode: bool = False,
        **kwargs,
    ):
        if debug_mode:
            additional_trainer_args = {
                "limit_train_batches": 10,
                "limit_val_batches": 10,
                "limit_test_batches": 10,
                "limit_predict_batches": 10,
            }
        else:
            additional_trainer_args = {}

        # best threshold on validation set
        val_trainer = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=NullLogger(),
            **additional_trainer_args,
        )  # type: ignore
        thresholds = np.round(np.linspace(0.05, 0.95, 19), decimals=3)
        metrics = {}
        for th in thresholds:
            metrics[f"f1-{th:.2f}"] = BinaryF1Score(threshold=th, ignore_index=-1)
            metrics[f"iou-{th:.2f}"] = BinaryJaccardIndex(threshold=th, ignore_index=-1)
            metrics[f"tgs-iou-{th:.2f}"] = BinaryTGSMeanIoU(
                pred_threshold=th, ignore_index=-1
            )

        model.metrics["val"] = metrics
        val_trainer.validate(
            model, datamodule=self.datamodule, ckpt_path=ckpt_path, verbose=False
        )

        f1 = {}
        iou = {}
        tgs_iou = {}
        for th in thresholds:
            f1[th] = float(val_trainer.callback_metrics[f"val_f1-{th:.2f}"].item())
            iou[th] = float(val_trainer.callback_metrics[f"val_iou-{th:.2f}"].item())
            tgs_iou[th] = float(
                val_trainer.callback_metrics[f"val_tgs-iou-{th:.2f}"].item()
            )
        f1_best_thr = max(f1, key=f1.get)  # type: ignore
        iou_best_thr = max(iou, key=iou.get)  # type: ignore
        tgs_iou_best_thr = max(tgs_iou, key=tgs_iou.get)  # type: ignore

        _log_dir = self._interpolate_eval_log_dir(logs_dir, execution_id, ckpt_name)
        _log_dir.mkdir(parents=True, exist_ok=True)
        th_search_path = _log_dir / "th_search.json"

        with open(th_search_path, "w") as f:
            json.dump(
                {
                    "search_space": list(thresholds),
                    "f1": {
                        "best": f1_best_thr,
                        "search": f1,
                    },
                    "iou": {
                        "best": iou_best_thr,
                        "search": iou,
                    },
                    "tgs_iou": {
                        "best": tgs_iou_best_thr,
                        "search": tgs_iou,
                    },
                },
                f,
                indent=4,
            )
        print(f"Threshold search results saved to: {th_search_path}")

        # HOw threshold should be selected? iou_best_thr, or tgs_iou_best_thr?
        # Should it be metric dependant? For now, we will select the one that maximizes standard iou (as it was), but we can change it to tgs_iou if we want to prioritize that metric....
        self.threshold = iou_best_thr
        print(
            f"Selected threshold: {self.threshold:.3f} (F1={f1_best_thr:.4f}, IoU={iou_best_thr:.4f}, TGS-IoU={tgs_iou_best_thr:.4f})"
        )
        print(f"Starting test evaluation with threshold: {self.threshold:.3f}...")
        model.metrics["test"] = self.metrics

        return super().eval(
            model=model,
            logs_dir=logs_dir,
            execution_id=execution_id,
            ckpt_path=ckpt_path,
            ckpt_name=ckpt_name,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            debug_mode=debug_mode,
            **kwargs,
        )


class RegressionTask(DeepRockTask):
    task_type = "regression"

    @property
    def loss(self):
        return loss.MSEWithSqueeze()

    @property
    def metrics(self):
        if self.datamodule is None:
            raise RuntimeError("'setup' method needs to be called first")
        metrics = {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "psnr": PeakSignalNoiseRatio(),
            "ssim": StructuralSimilarityIndexMeasure(),
        }
        return metrics

    def eval(
        self,
        model: DeepRockModel,
        logs_dir: Path,
        execution_id: int,
        ckpt_path: Path,
        ckpt_name: str,
        accelerator: str = "gpu",
        devices: Union[str, int, List[int]] = 1,
        precision: str = "32-true",
        debug_mode: bool = False,
        **kwargs,
    ):
        return super().eval(
            model=model,
            logs_dir=logs_dir,
            execution_id=execution_id,
            ckpt_path=ckpt_path,
            ckpt_name=ckpt_name,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            debug_mode=debug_mode,
            **kwargs,
        )


class SegmentationTask(DeepRockTask):
    task_type = "multiclass"

    @property
    def loss(self):
        return loss.CrossEntropy()

    @property
    def metrics(self):
        if self.datamodule is None:
            raise RuntimeError("'setup' method needs to be called first")

        mious = {
            f"miou-{average}": JaccardIndex(
                num_classes=self.datamodule.out_channels,
                task="multiclass",
                average=average,  # type: ignore
                ignore_index=-1,
            )
            for average in ["macro", "micro", "weighted"]
        }

        f1_scores = {
            f"f1-{average}": F1Score(
                num_classes=self.datamodule.out_channels,
                task="multiclass",
                average=average,  # type: ignore
                ignore_index=-1,
            )
            for average in ["macro", "micro", "weighted"]
        }
        return {**mious, **f1_scores}


class NullLogger(Logger):
    @property
    def name(self):
        return "null"

    @property
    def version(self):
        return "0"

    @property
    def experiment(self):
        return None

    def log_metrics(self, metrics, step):
        pass

    def log_hyperparams(self, params):
        pass