from pathlib import Path
from typing import Literal, Optional, Tuple, Any, List
from collections import defaultdict
from itertools import cycle
import torch

import numpy as np
import lightning as L
from torch.utils.data import DataLoader

from minerva.data.datasets.base import SimpleDataset
from deeprock.datamodules.subsets import RandomSubset, BinaryTreeSubset, FirstNSubset


class DeepRockDataset(SimpleDataset):
    def __init__(
        self,
        stage: Literal["train", "val", "test"],
        crop_size: Optional[Tuple[int, int]] = None,
        model_resolution: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
        norm: Optional[Literal["znorm", "clipnorm"]] = None,
        domain_norm: List[Tuple[int, int]] = None,
        channels: int = 3,
        seed: int = 42,
        data_transforms: Any = None,
        label_transforms: Any = None,
        task_type: str = Literal["regression", "binary", "multiclass"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        if target_size is not None and crop_size is not None:
            target_size = None
        self.current_epoch = 0
        self.stage = stage
        self.seed = seed
        self.crop_size = crop_size
        self.model_resolution = model_resolution
        self.target_size = target_size
        self.norm = norm
        self.domain_norm = domain_norm
        self.norm_label = False
        self.channels = channels
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms
        self.task_type = task_type

    def set_epoch(self, idx: int):
        self.current_epoch = idx

    def _random_crop(
        self, items: Tuple[np.ndarray, np.ndarray], seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        generator = np.random.default_rng(seed=seed)
        shape = items[0].shape
        starting_point = [
            int(generator.random() * (s - p + 1)) for p, s in zip(self.crop_size, shape)
        ]
        sl = tuple([slice(s, s + p) for s, p in zip(starting_point, self.crop_size)])
        items = tuple([item[sl] for item in items])
        return items

    def _sliding_window(self, data: np.ndarray) -> Tuple[np.ndarray, list]:
        strides = np.array(self.crop_size) // 2
        starting_points = []
        indexes = tuple(
            (
                np.ceil((np.array(data.shape) - np.array(self.crop_size)) / strides)
            ).astype(int)
            + 1
        )
        patches = []
        for patch_index in np.ndindex(indexes):
            point = tuple(
                [
                    idx * stride if idx * stride + patch_len < ma else ma - patch_len
                    for idx, stride, patch_len, ma in zip(
                        patch_index, strides, self.crop_size, data.shape
                    )
                ]
            )
            sl = [
                slice(p, p + patch_len, None)
                for p, patch_len in zip(point, self.crop_size)
            ]
            starting_points.append(point)
            patches.append(data[tuple(sl)])

        return np.stack(patches), starting_points

    def _norm(self, data: np.ndarray) -> np.ndarray:
        eps = 1e-8
        mean = data.mean(axis=(-2, -1), keepdims=True)
        std = data.std(axis=(-2, -1), keepdims=True) + eps
        znorm = (data - mean) / std
        if self.norm == "znorm" or self.domain_norm is not None:
            return znorm
        clipped = np.clip(znorm, -10, 10)
        ma = np.max(clipped, axis=(-2, -1), keepdims=True)
        mi = np.min(clipped, axis=(-2, -1), keepdims=True)

        x = (clipped - mi) / ((ma - mi) + eps)
        return x

    def _domain_norm(self, data: np.ndarray) -> np.ndarray:
        for i, (channel, channel_norm) in enumerate(zip(data, cycle(self.domain_norm))):
            data[i] = channel * channel_norm[1] + channel_norm[0]
        if self.norm == "clipnorm":
            data = np.clip(data, 0, 1)
        return data

    def _pad(
        self, data: np.ndarray, value: Optional[float] = None
    ) -> Tuple[np.ndarray, Tuple[slice, slice]]:
        if data.ndim == 2:
            h, w = data.shape
            c_first = False
        elif data.ndim == 3:
            c_first = True
            _, h, w = data.shape
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")
        model_resolution = 1 if self.model_resolution is None else self.model_resolution
        if self.target_size is not None:
            pad_h = int(
                np.ceil(self.target_size[0] / model_resolution) * model_resolution - h
            )
            pad_w = int(
                np.ceil(self.target_size[1] / model_resolution) * model_resolution - w
            )
        else:
            pad_h = int(np.ceil(h / model_resolution) * model_resolution - h)
            pad_w = int(np.ceil(w / model_resolution) * model_resolution - w)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        sl = [(0, h), (0, w)]

        if pad_h == 0 and pad_w == 0:
            return data, sl

        pad_width = (
            ((pad_top, pad_bottom), (pad_left, pad_right))
            if not c_first
            else ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        )

        sl = [(pad_top, pad_top + h), (pad_left, pad_left + w)]

        data = (
            np.pad(data, pad_width, mode="reflect")
            if value is None
            else np.pad(data, pad_width, mode="constant", constant_values=value)
        )
        return data, sl

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        items = super().__getitem__(idx)
        meta = {}
        if (
            self.crop_size is not None
        ):  # Model will be trained with RandomCrop and prediction will use a Sliding Window
            if self.stage == "train":
                seed = (self.current_epoch + self.seed) * len(self) + idx
                items = self._random_crop(items=items, seed=seed)
            else:
                patched_data, points = self._sliding_window(items[0])
                items = (
                    patched_data,
                    items[1],
                )
                meta["points"] = points
        if self.norm is not None:
            items = (
                self._norm(items[0]),
                self._norm(items[1]) if self.norm_label else items[1],
            )
        if (
            self.model_resolution is not None or self.target_size is not None
        ):  # Pad input to match model resolution (input%model_resolution = 0)
            padded_item_0, pad = self._pad(items[0])  # or to match target_size
            padded_item_1, pad_label = (
                self._pad(items[1], value=-1)
                if self.task_type != "regression"
                else self._pad(items[1])
            )
            meta["pad"] = pad
            meta["pad_label"] = pad_label
            items = (padded_item_0, padded_item_1)

        if self.channels is not None:
            items = (
                np.repeat(
                    np.expand_dims(items[0], axis=-3), repeats=self.channels, axis=-3
                ),
                items[1],
            )
        if self.domain_norm is not None:
            items = (self._domain_norm(items[0]), items[1])

        return (
            (
                self.data_transforms(items[0])
                if self.data_transforms is not None
                else items[0]
            ),
            (
                self.label_transforms(items[1])
                if self.label_transforms is not None
                else items[1]
            ),
            meta,
        )


class DeepRockDataModule(L.LightningDataModule):
    AVAILABLE_DATASETS = None
    norm_label = False
    task = None

    def __init__(
        self,
        root_dir: Path,
        dataset: str,
        crop_size: Optional[Tuple[int, int]] = None,
        model_resolution: Optional[int] = None,
        norm: Optional[Literal["znorm", "clipnorm"]] = None,
        domain_norm: List[Tuple[int, int]] = None,
        channels: int = 3,
        partition: Optional[str] = None,
        batch_size: int = 8,
        eval_batch_size: int = 1,
        seed: int = 42,
        task_type: str = Literal["regression", "binary", "multiclass"],
        train_subset_type: Optional[Literal["random", "binary_tree", "first_n"]] = None,
        train_subset_size: Optional[int] = None,
        train_subset_kwargs: Optional[dict] = None,
        drop_last_train: bool = True,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.dataset = self.check_dataset(dataset)
        self.crop_size = crop_size
        self.model_resolution = model_resolution
        self.norm = norm
        self.domain_norm = domain_norm
        self.channels = channels
        self.partition = partition
        self.seed = seed
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.task_type = task_type
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_subset_type = train_subset_type
        self.train_subset_size = train_subset_size
        self.train_subset_kwargs = train_subset_kwargs or {}
        self.drop_last_train = drop_last_train
        if self.train_subset_type is not None:
            if self.train_subset_size is None:
                raise ValueError(
                    "subset_size must be specified when subset_type is provided."
                )
            if self.train_subset_size <= 0:
                raise ValueError("subset_size must be a positive integer.")

    def setup(self, stage: str = None):
        self.train_dataset = self.get_dataset("train")
        if self.train_subset_type is not None:
            self.train_dataset = self.get_subset(self.train_dataset)
        self.val_dataset = self.get_dataset("val")
        self.test_dataset = self.get_dataset("test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=torch.get_num_threads(),
            collate_fn=collate_fn,
            drop_last=self.drop_last_train,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size if self.val_dataset.crop_size is None else 1,
            num_workers=torch.get_num_threads(),
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=torch.get_num_threads(),
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False,
        )

    def get_subset(self, dataset: DeepRockDataset) -> DeepRockDataset:
        if self.train_subset_type == "random":
            return RandomSubset(
                dataset,
                n_elements=self.train_subset_size,
                # seed=self.seed,
                **self.train_subset_kwargs,
            )
        elif self.train_subset_type == "binary_tree":
            return BinaryTreeSubset(
                dataset, n_elements=self.train_subset_size, **self.train_subset_kwargs
            )
        elif self.train_subset_type == "first_n":
            return FirstNSubset(
                dataset, n_elements=self.train_subset_size, **self.train_subset_kwargs
            )
        else:
            raise ValueError(
                f"Invalid subset_type '{self.train_subset_type}'. Must be 'random', 'binary_tree', or 'first_n'."
            )

    def get_dataset(self, stage: Literal["train", "test", "val"]) -> DeepRockDataset:
        dataset_config = self.AVAILABLE_DATASETS[self.dataset]
        partition = (
            self.partition
            if self.partition is not None
            else dataset_config["partition"]
        )
        readers = [
            dataset_config["readers"][stage][0][0](
                self.root_dir
                / "processed"
                / self.task
                / self.dataset
                / partition
                / stage
                / self.data_relpath,
                **dataset_config["readers"][stage][0][1],
            ),
            dataset_config["readers"][stage][1][0](
                self.root_dir
                / "processed"
                / self.task
                / self.dataset
                / partition
                / stage
                / self.label_relpath,
                **dataset_config["readers"][stage][1][1],
            ),
        ]
        return DeepRockDataset(
            stage=stage,
            crop_size=self.crop_size,
            model_resolution=self.model_resolution,
            target_size=dataset_config.get("target_size"),
            norm=self.norm,
            domain_norm=self.domain_norm,
            channels=self.channels,
            seed=self.seed,
            readers=readers,
            data_transforms=dataset_config["data_transforms"],
            label_transforms=dataset_config["label_transforms"],
            task_type=self.task_type,
        )

    @classmethod
    def check_dataset(cls, dataset: str):
        if cls.AVAILABLE_DATASETS is None:
            raise NotImplementedError
        if dataset not in cls.AVAILABLE_DATASETS:
            raise ValueError(
                f"Invalid dataset! Available datasets are: {', '.join(cls.AVAILABLE_DATASETS)}"
            )
        return dataset

    @property
    def data_relpath(self) -> Path:
        return Path("data")

    @property
    def label_relpath(self) -> Path:
        return Path("label")

    @property
    def out_channels(self):
        dataset_config = self.AVAILABLE_DATASETS[self.dataset]
        return dataset_config["out_channels"]

    @property
    def output_shape(self):
        dataset_config = self.AVAILABLE_DATASETS[self.dataset]
        return dataset_config.get("output_shape")


def collate_fn(samples_list):
    data = []
    label = []
    metatadata = defaultdict(list)
    for sample in samples_list:
        data.append(torch.from_numpy(sample[0]))
        label.append(torch.from_numpy(sample[1]))
        for k, v in sample[2].items():
            metatadata[k].append(v)

    return (torch.stack(data), torch.stack(label), metatadata)
