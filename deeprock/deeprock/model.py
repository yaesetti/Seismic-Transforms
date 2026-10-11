from typing import Any, Dict, Optional, Union, Callable, Tuple, Literal

import lightning as L

import torch
from torchmetrics import Metric, MeanMetric


class DeepRockModel(L.LightningModule):

    def __init__(
        self,
        backbone: torch.nn.Module,
        prediction_head: torch.nn.Module,
        loss_fn: torch.nn.Module,
        task_type: Literal["binary", "regression", "segmentation"],
        adapter: Optional[
            Callable[[torch.nn.Module, torch.Tensor, str], torch.Tensor]
        ] = None,
        learning_rate: Tuple[float, float] = (1e-3, 1e-3),
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
        optimizer: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[type] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initializes the supervised model with training components and configs.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone (feature extractor) model.
        prediction_head : torch.nn.Module
            The fully connected head. Use nn.Identity() if not required.
        loss_fn : torch.nn.Module
            Loss function to optimize during training.
        adapter : Callable, optional
            Function to transform backbone outputs before feeding into `fc`.
        learning_rate : float, default=1e-3
            Learning rate used for optimization.
        flatten : bool, default=True
            If True, flattens backbone outputs before `fc`.
        train_metrics : dict, optional
            TorchMetrics dictionary for training evaluation.
        val_metrics : dict, optional
            TorchMetrics dictionary for validation evaluation.
        test_metrics : dict, optional
            TorchMetrics dictionary for test evaluation.
        freeze_backbone : bool, default=False
            If True, backbone parameters are frozen during training.
        optimizer: type
            Optimizer class to be instantiated. By default, it is set to
            `torch.optim.Adam`. Should be a subclass of
            `torch.optim.Optimizer` (e.g., `torch.optim.SGD`).
        optimizer_kwargs : dict, optional
            Additional kwargs passed to the optimizer constructor.
        lr_scheduler : type, optional
            Learning rate scheduler class to be instantiated. By default, it is
            set to None, which means no scheduler will be used. Should be a
            subclass of `torch.optim.lr_scheduler.LRScheduler` (e.g.,
            `torch.optim.lr_scheduler.StepLR`).
        lr_scheduler_kwargs : dict, optional
            Additional kwargs passed to the scheduler constructor.
        """

        super().__init__()
        self.backbone = backbone
        self.prediction_head = prediction_head
        self.loss_fn = loss_fn
        self.adapter = adapter
        self.learning_rate = learning_rate
        self.task_type = task_type
        self.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}
        self.running_train_loss = MeanMetric()

        self._set_trainable_params()

    def _set_trainable_params(self):
        """Freeze the parameters of the backbone model if `freeze_backbone` is
        set to True."""
        if hasattr(self.backbone, "parameters"):
            for param in self.backbone.parameters():
                param.requires_grad = self.learning_rate[0] != 0

        if hasattr(self.prediction_head, "parameters"):
            for param in self.prediction_head.parameters():
                param.requires_grad = self.learning_rate[1] != 0

    def _loss_func(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between the output and the input data.

        Parameters
        ----------
        y_hat : torch.Tensor
            The output data from the forward pass.
        y : torch.Tensor
            The input data/label.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        loss = self.loss_fn(y_hat, y)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass with the input data on the backbone model.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The output data from the forward pass.
        """
        dims = x.shape[-2:]  # H, W
        x = (
            self.backbone(x)
            if self.adapter is None
            else self.adapter(self.backbone, x, self.task_type)
        )
        x = self.prediction_head(x, dims)
        return x

    def _compute_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, step_name: str
    ) -> Dict[str, torch.Tensor]:
        """Calculate the metrics for the given step.

        Parameters
        ----------
        y_hat : torch.Tensor
            The output data from the forward pass.
        y : torch.Tensor
            The input data/label.
        step_name : str
            Name of the step. It will be used to get the metrics from the
            `self.metrics` attribute.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary with the metrics values.
        """
        if self.metrics[step_name] is None:
            return {}

        return {
            f"{step_name}_{metric_name}": metric.to(self.device)(y_hat, y)
            for metric_name, metric in self.metrics[step_name].items()
        }

    def _epoch_metrics(self, step_name: str) -> Dict[str, torch.Tensor]:
        """Calculate the metrics for the epoch and resets metrics.

        Parameters
        ----------
        step_name : str
            Name of the step. It will be used to get the metrics from the
            `self.metrics` attribute.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary with the metrics values.
        """
        if self.metrics.get(step_name) is None:
            return {}

        epoch_metrics = {
            f"epoch_{step_name}_{metric_name}": metric.compute()
            for metric_name, metric in self.metrics[step_name].items()
        }

        for metric in self.metrics[step_name].values():
            metric.reset()

        if step_name == "test":
            for metric_name, metric_value in epoch_metrics.items():
                self.log(
                    metric_name,
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )

    def _remove_padding(
        self, data: torch.Tensor, pad: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> torch.Tensor:
        sl = [slice(None)] * data.ndim
        for i, p in enumerate(pad):
            sl[i - 2] = slice(p[0], p[1])
        return data[tuple(sl)]

    def _activation_function(self, data: torch.Tensor):
        if self.task_type == "binary":
            return torch.sigmoid(data)
        elif self.task_type == "multiclass":
            return torch.softmax(data, dim=1)
        return data

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        """Perform a single train/validation/test step. It consists in making a
        forward pass with the input data on the backbone model, computing the
        loss between the output and the input data, and logging the loss.

        Parameters
        ----------
        batch : torch.Tensor
            The input data. It must be a 2-element tuple of tensors, where the
            first tensor is the input data and the second tensor is the mask.
        batch_idx : int
            The index of the batch.
        step_name : str
            The name of the step. It will be used to log the loss. The possible
            values are: "train", "val" and "test". The loss will be logged as
            "{step_name}_loss".

        Returns
        -------
        torch.Tensor
            A tensor with the loss value.
        """
        x, y, metadata = batch
        if x.ndim == 5:  # patch inference
            x = x.squeeze(0)
        y_hat = self.forward(x)

        if "points" in metadata:  # Patch Inference
            y_hat = (
                self._remove_padding(y_hat, metadata["pad"][0])
                if "pad" in metadata
                else y_hat
            )
            y = (
                self._remove_padding(y, metadata["pad_label"][0])
                if "pad_label" in metadata
                else y
            )
            y_hat_full = torch.zeros(
                y_hat.shape[1], *y.shape[-2:], dtype=y_hat.dtype, device=y_hat.device
            )
            weights = torch.zeros_like(y_hat_full)
            y_hat = self._activation_function(y_hat)
            for points, part in zip(metadata["points"][0], y_hat):
                sl = tuple([slice(p, p + s) for p, s in zip([0, *points], part.shape)])
                y_hat_full[sl] += part
                weights[sl] += 1

            y_hat = y_hat_full / weights
            y_hat = (
                y_hat.unsqueeze(0) if y_hat.ndim == 3 else y_hat
            )  # unsqueeze batch_dim

        if self.task_type in ["regression", "binary"]:  # check if it is needed
            y = y.unsqueeze(1) if y.ndim == 3 else y
        loss = self._loss_func(y_hat, y)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # Add sync_dist parameter and set always true
            batch_size=len(y)
        )

        if step_name == "train":
            self.running_train_loss(loss)

        metrics = self._compute_metrics(y_hat, y, step_name)
        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=step_name == "test",
                on_epoch=step_name != "test",
                prog_bar=False,
                logger=True,
                sync_dist=True,  # Add sync_dist parameter and set always true
                batch_size=len(y),
            )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, step_name="test")

    def on_train_epoch_end(self):
        self._epoch_metrics("train")
        self.running_train_loss.reset()

    def on_validation_epoch_end(self):
        self._epoch_metrics("val")
        current_avg_loss = self.running_train_loss.compute()
        self.log("train_loss", current_avg_loss)

    def on_test_epoch_end(self):
        self._epoch_metrics("test")

    def configure_optimizers(self):
        self._set_trainable_params()

        param_groups = [
            {"params": self.backbone.parameters(), "lr": self.learning_rate[0]},
            {"params": self.prediction_head.parameters(), "lr": self.learning_rate[1]},
        ]

        optimizer = self.optimizer(param_groups, **self.optimizer_kwargs)
        if self.lr_scheduler is None:
            return optimizer

        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

        return [optimizer], [scheduler]
