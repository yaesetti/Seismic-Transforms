from typing import List
import numpy as np
from minerva.transforms.transform import _Transform


class InstanceNormalization(_Transform):
    def __init__(self, eps: float = 1e-5, axis: tuple = (1, 2)):
        self.eps = eps
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply z-normalization (zero mean, unit variance) per channel.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (C, H, W). Each channel will be normalized
            independently by subtracting its mean and dividing by its standard deviation.

        Returns
        -------
        np.ndarray
            Normalized array with the same shape as the input.

        Notes
        -----
        - Mean and standard deviation are computed over the spatial dimensions (H, W)
          for each channel.
        - A small epsilon (1e-5) is added to the standard deviation to prevent division by zero.
        """
        mean = x.mean(axis=self.axis, keepdims=True)
        std = x.std(axis=self.axis, keepdims=True) + self.eps
        return (x - mean) / std

    def __str__(self):
        return "InstanceNormalization()"


class InstanceNormalization2(InstanceNormalization):
    def __init__(self, eps: float = 1e-5, clip_value: float = 5, axis: tuple = (1, 2)):
        super().__init__(eps=eps, axis=axis)
        self.clip_value = clip_value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply z-normalization (zero mean, unit variance) per channel, followed by clipping and then MinMax normalization.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (C, H, W). Each channel will be normalized
            independently by subtracting its mean and dividing by its standard deviation.

        Returns
        -------
        np.ndarray
            Normalized array with the same shape as the input.

        Notes
        -----
        - Mean and standard deviation are computed over the spatial dimensions (H, W)
          for each channel.
        - A small epsilon (1e-5) is added to the standard deviation to prevent division by zero.
        """
        znorm = super().__call__(x)
        clipped = np.clip(znorm, -self.clip_value, self.clip_value)
        ma = np.max(clipped)
        mi = np.min(clipped)
        return (clipped - mi) / (ma - mi)

    def __str__(self):
        return "InstanceNormalization2()"


class BinarizeLabel(_Transform):
    """Transform any label different than 0 to 1."""

    def __init__(self, cast_to: str = "int64"):
        self.cast_to = cast_to

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Binarize the input array by setting all non-zero values to 1.

        Parameters
        ----------
        x : np.ndarray
            Input array of any shape. All non-zero values will be set to 1.

        Returns
        -------
        np.ndarray
            Binarized array with the same shape as the input, where all non-zero values are replaced with 1.
        """
        return (x != 0).astype(self.cast_to)

    def __str__(self):
        return "BinarizeLabel()"
