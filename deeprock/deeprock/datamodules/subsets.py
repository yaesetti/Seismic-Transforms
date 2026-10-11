from typing import Optional

import torch
from torch.utils.data import Subset
from math import ceil, floor


def build_indices(start: int, end: int, n_elements: int) -> list[int]:
    """
    Build a list of indices using binary tree traversal.

    Parameters
    ----------
    start : int
        The starting index (inclusive).
    end : int
        The ending index (exclusive).
    n_elements : int
        The number of elements to select.

    Returns
    -------
    list[int]
        A list of indices selected using binary tree traversal.
    """
    if (end - start <= 0) or (n_elements <= 0):
        return []

    midpoint = (end + start) // 2
    r = [midpoint]

    remainder = n_elements - 1
    left_apportion = ceil(remainder / 2)
    right_apportion = floor(remainder / 2)

    r += build_indices(start, midpoint, left_apportion)
    r += build_indices(midpoint + 1, end, right_apportion)

    return r


class BinaryTreeSubset(Subset):
    def __init__(
        self, dataset: torch.utils.data.Dataset, n_elements: int, sort: bool = True
    ):
        """Subset selecting items by recursively traversing a binary tree of indices.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sample from.
        n_elements : int
            Number of elements to include in the subset.
        sort : bool, default=True
            Whether to sort the selected indices before constructing the subset.
        """
        indices = build_indices(0, len(dataset), n_elements)
        if len(indices) != n_elements:
            raise ValueError(
                f"Cannot build a binary tree subset with {n_elements} elements from a dataset of size {len(dataset)}"
            )
        if sort:
            indices.sort()
        super().__init__(dataset, indices)


class RandomSubset(Subset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_elements: int,
        sort: bool = True,
        seed: Optional[int] = None,
    ):
        """Subset selecting random indices from a dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sample from.
        n_elements : int
            Number of elements to include in the subset.
        sort : bool, default=True
            Whether to sort the selected indices before constructing the subset.
        seed : Optional[int], default=None
            Seed for the random number generator.
        """
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=rng)[:n_elements].tolist()
        if len(indices) != n_elements:
            raise ValueError(
                f"Cannot build a random subset with {n_elements} elements from a dataset of size {len(dataset)}"
            )
        if sort:
            indices.sort()
        super().__init__(dataset, indices)


class FirstNSubset(Subset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_elements: int,
        sort: bool = True,
        reverse: bool = False,
    ):
        """Subset selecting the first or last ``n_elements`` from a dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sample from.
        n_elements : int
            Number of elements to include in the subset.
        sort : bool, default=True
            Whether to sort the selected indices before constructing the subset.
        reverse : bool, default=False
            If ``True``, select the last ``n_elements`` in reverse order.
        """
        if len(dataset) < n_elements:
            raise ValueError(
                f"Cannot build a first N subset with {n_elements} elements from a dataset of size {len(dataset)}"
            )

        if reverse:
            indices = list(range(len(dataset) - 1, len(dataset) - n_elements - 1, -1))
        else:
            indices = list(range(n_elements))

        if sort:
            indices.sort()
        super().__init__(dataset, indices)


class StratifiedSubset(Subset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        targets: list[int],
        n_elements: int,
        subset_cls: type = RandomSubset,
        sort: bool = True,
        ignore_target: int = -1,
    ):
        """Subset sampling elements in a stratified manner across target groups.

        Given a dataset and a list of targets for each sample of the dataset, this class:
        1. Groups the samples by their target value.
        2. For each group, it creates a subset of the samples using the provided subset class (default is RandomSubset).
        3. Combines all the subsets into a single subset, ensuring that the total number of samples is equal to n_elements.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sample from.
        targets : list[int]
            Target value for each sample in ``dataset``.
        n_elements : int
            Number of elements to include in the subset.
        subset_cls : type, default=RandomSubset
            Subset class used to sample within each target group.
        sort : bool, default=True
            Whether to sort the selected indices before constructing the subset.
        ignore_target : int, default=-1
            Target value to exclude from stratification.

        Raises
        -----
        ValueError
            If the length of ``dataset`` and ``targets`` do not match.
            If there are no valid target groups found for stratification.
            If ``n_elements`` is less than the number of target groups.
            If the total number of selected indices does not equal ``n_elements``.
        """

        if len(dataset) != len(targets):
            raise ValueError(
                f"Length of dataset ({len(dataset)}) and targets ({len(targets)}) must be the same."
            )

        # Group indices by target value
        target_to_indices = {}
        for idx, target in enumerate(targets):
            if target == ignore_target:
                continue
            if target not in target_to_indices:
                target_to_indices[target] = []
            target_to_indices[target].append(idx)

        # Calculate the number of elements to sample from each group
        n_groups = len(target_to_indices)
        if n_groups == 0:
            raise ValueError("No valid target groups found for stratification.")
        if n_elements < n_groups:
            raise ValueError(
                f"Cannot build a stratified subset with {n_elements} elements from {n_groups} target groups."
            )
        n_per_group = n_elements // n_groups
        remainder = n_elements % n_groups

        indices = []
        for i, (target, group_indices) in enumerate(target_to_indices.items()):
            n_samples = n_per_group + (1 if i < remainder else 0)
            subset = subset_cls(dataset, n_samples, sort=False)
            selected_indices = [group_indices[idx] for idx in subset.indices]
            indices.extend(selected_indices)

        if len(indices) != n_elements:
            raise ValueError(
                f"Cannot build a stratified subset with {n_elements} elements from a dataset of size {len(dataset)}"
            )
        if sort:
            indices.sort()
        super().__init__(dataset, indices)
