import torch
from torch.utils.data import Subset
from math import ceil, floor

import pytest
from deeprock.datamodules.subsets import (
    build_indices,
    BinaryTreeSubset,
    RandomSubset,
    FirstNSubset,
    StratifiedSubset,
)


class _DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self._data = list(range(size))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def test_build_indices_handles_empty_and_non_positive_requests():
    assert build_indices(0, 0, 1) == []
    assert build_indices(2, 1, 1) == []
    assert build_indices(0, 5, 0) == []


def test_build_indices_uses_binary_tree_traversal():
    assert build_indices(0, 7, 7) == [3, 1, 0, 2, 5, 4, 6]


def test_binary_tree_subset_sorts_and_validates_size():
    dataset = _DummyDataset(7)
    subset = BinaryTreeSubset(dataset, 3, sort=False)
    assert subset.indices == [3, 1, 5]

    sorted_subset = BinaryTreeSubset(dataset, 3, sort=True)
    assert sorted_subset.indices == [1, 3, 5]


def test_binary_tree_subset_raises_for_impossible_request():
    dataset = _DummyDataset(2)
    with pytest.raises(ValueError):
        BinaryTreeSubset(dataset, 3)


def test_random_subset_sorts_and_validates_size():
    dataset = _DummyDataset(10)
    subset = RandomSubset(dataset, 4, sort=True)
    assert len(subset.indices) == 4
    assert subset.indices == sorted(subset.indices)
    assert len(set(subset.indices)) == 4


def test_random_subset_raises_for_impossible_request():
    dataset = _DummyDataset(2)
    with pytest.raises(ValueError):
        RandomSubset(dataset, 3)


def test_first_n_subset_selects_first_and_last_items():
    dataset = _DummyDataset(5)
    assert FirstNSubset(dataset, 3, sort=False).indices == [0, 1, 2]
    assert FirstNSubset(dataset, 3, reverse=True, sort=False).indices == [4, 3, 2]


def test_first_n_subset_raises_for_impossible_request():
    dataset = _DummyDataset(2)
    with pytest.raises(ValueError):
        FirstNSubset(dataset, 3)


def test_stratified_subset_validates_lengths():
    dataset = _DummyDataset(4)
    targets = [0, 0, 1, 1]
    subset = StratifiedSubset(dataset, targets, 2, subset_cls=FirstNSubset, sort=True)
    assert len(subset.indices) == 2
    assert subset.indices == [0, 2]

    with pytest.raises(ValueError):
        StratifiedSubset(dataset, targets[:-1], 2)


def test_stratified_subset_invalid_number_of_groups():
    dataset = _DummyDataset(6)
    targets = [0, 0, 1, 1, 2, 2]

    with pytest.raises(ValueError):
        StratifiedSubset(dataset, targets, 2, subset_cls=FirstNSubset, sort=True)
