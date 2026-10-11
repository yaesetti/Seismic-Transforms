import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple, Literal
import os
import json
from minerva.utils.typing import PathLike
from minerva.data.readers import _Reader
from minerva.data.readers.base_file_iterator import BaseFileIterator


class NumpyFolderReader(BaseFileIterator):
    def __init__(
        self,
        path: PathLike,
        sort_method: Optional[List[str]] = None,
        delimiter: Optional[str] = None,
        key_index: Union[int, List[int]] = 0,
        reverse: bool = False,
        allow_pickle: bool = True,
        array_key: Optional[str] = None,
    ):
        """Load image files from a directory.

        Parameters
        ----------
        path : Union[Path, str]
            The path to the directory containing the image files. Files will be
            searched recursively.
        sort_method : Optional[List[str]], optional
            A list specifying how to sort each part of the filename. Each
            element can  be either "text" (lexicographical) or "numeric"
            (numerically). By default, None, which will use "numeric" if
            numeric parts are detected.
        delimiter : Optional[str], optional
            The delimiter to split filenames into components, by default None.
        key_index : Union[int, List[int]], optional
            The index (or list of indices) of the part(s) of the filename to
            use  for sorting. If a list is provided, files will be sorted
            based on  multiple parts in sequence. Thus, first by the part at
            index 0, then by the part at index 1, and so on. By default 0.
        reverse : bool, optional
            Whether to sort in reverse order, by default False.
        filters: Optional[Union[List[str], str]]
            An optional string or list of strings containing regular expressions
            with which to filter files by their stems. Files that match at least
            one pattern are kept, and the others are excluded. Defaults to None,
            which means no files are excluded.

        Raises
        ------
        NotADirectoryError
            If the path is not a directory.
        """
        self.root_dir = Path(path)
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"{path} is not a directory.")

        files = list(self.root_dir.rglob("*.npy")) + list(self.root_dir.rglob("*.npz"))
        self.allow_pickle = allow_pickle
        self.array_key = array_key
        super().__init__(files, sort_method, delimiter, key_index, reverse)

    def __getitem__(self, index: int) -> np.ndarray:
        """Retrieve file at the specified index."""
        p = self.files[index].as_posix()
        if self.files[index].suffix == ".npz":
            return np.load(p, allow_pickle=self.allow_pickle)[self.array_key]
        else:
            return np.load(p, allow_pickle=self.allow_pickle)

    def __str__(self) -> str:
        return f"NumpyFolderReader at '{self.root_dir}' ({len(self.files)} files)"


class TextReader(BaseFileIterator):
    def __init__(
        self,
        path: PathLike,
        sort_method: Optional[List[str]] = None,
        delimiter: Optional[str] = None,
        key_index: Union[int, List[int]] = 0,
        reverse: bool = False,
    ):
        """Load image files from a directory.

        Parameters
        ----------
        path : Union[Path, str]
            The path to the directory containing the image files. Files will be
            searched recursively.
        sort_method : Optional[List[str]], optional
            A list specifying how to sort each part of the filename. Each
            element can  be either "text" (lexicographical) or "numeric"
            (numerically). By default, None, which will use "numeric" if
            numeric parts are detected.
        delimiter : Optional[str], optional
            The delimiter to split filenames into components, by default None.
        key_index : Union[int, List[int]], optional
            The index (or list of indices) of the part(s) of the filename to
            use  for sorting. If a list is provided, files will be sorted
            based on  multiple parts in sequence. Thus, first by the part at
            index 0, then by the part at index 1, and so on. By default 0.
        reverse : bool, optional
            Whether to sort in reverse order, by default False.
        filters: Optional[Union[List[str], str]]
            An optional string or list of strings containing regular expressions
            with which to filter files by their stems. Files that match at least
            one pattern are kept, and the others are excluded. Defaults to None,
            which means no files are excluded.

        Raises
        ------
        NotADirectoryError
            If the path is not a directory.
        """
        self.root_dir = Path(path)
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"{path} is not a directory.")

        files = list(self.root_dir.rglob("*.txt"))
        super().__init__(files, sort_method, delimiter, key_index, reverse)

    def __getitem__(self, index: int) -> np.ndarray:
        """Retrieve the TIFF file at the specified index."""
        p = self.files[index].as_posix()
        with open(p, "r") as f:
            data = f.read()
        return np.array(data)

    def __str__(self) -> str:
        return f"TextReader at '{self.root_dir}' ({len(self.files)} files)"


class NumpyPatchReader(_Reader):
    def __init__(
        self,
        path: PathLike,
        patch_size: Optional[Tuple[int, int]] = None,
        patch_stride: Optional[Tuple[int, int]] = None,
        norm: Optional[Literal["min-max", "z-score"]] = None,
        crop_h: Optional[Tuple[int, int]] = None,
        crop_w: Optional[Tuple[int, int]] = None,
    ):
        """Load patches from a structured directory like 'path/{panel}/{rows}_{cols}.npy'.
        If patch_size is provided, each file will be treated as a grid of patches of the specified size,
        and __getitem__ will return the appropriate patch based on the index. For this case {rows}_{cols}.npy
        must reflect the dimensions of the given panel.
        If norm is provided, each patch will be normalized using the specified method, with statistics read
        from a .stats.json file located in the same directory as the .npy files.

        Parameters
        ----------
        path : Union[Path, str]
            The path to the directory containing the image files. Files will be
            searched recursively.
        patch_size : Optional[Tuple[int, int]], optional
            If provided, the reader will treat each file as a grid of patches of the specified size.
            The total number of patches will be calculated accordingly, and __getitem__ will return
            the appropriate patch based on the index. By default None, which means each file is treated as a single patch.
        patch_stride : Optional[Tuple[int, int]], optional
            If provided, the reader will use the specified stride when dividing files into patches.
            By default it will be set equal to patch_size, meaning non-overlapping patches.
        norm : Optional[Literal["min-max", "z-score"]], optional
            If provided, the reader will normalize each patch using the specified method.
                The required statistics for normalization (mean and std for z-score, min and max for min-max)
                will be read from a .stats.json file located in the same directory as the .npy files.
                By default None, which means no normalization is applied.
        crop_h : Optional[Tuple[int, int]], optional
            If provided, the reader will crop each file along the height dimension before extracting patches.
            The tuple should specify the start and end indices for cropping. By default None, which means no cropping is applied.
        crop_w : Optional[Tuple[int, int]], optional
            If provided, the reader will crop each file along the width dimension before extracting patches.
            The tuple should specify the start and end indices for cropping. By default None, which means no cropping is applied.
        Raises
        ------
        NotADirectoryError
            If the path is not a directory.
        """
        self.root_dir = Path(path)
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"{path} is not a directory.")

        if (self.root_dir / ".index.json").is_file():
            with open(self.root_dir / ".index.json", "r") as f:
                self.files = [self.root_dir / file for file in json.load(f)]
        else:
            self.files = sorted(list(self.root_dir.rglob("*.npy")))

        self.norm = norm
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.crop_h = crop_h
        self.crop_w = crop_w
        patches = None
        if self.patch_size is not None:
            patches = []
            for f in self.files:
                rows, cols = f.stem.split("_")
                rows, cols = int(rows), int(cols)
                if self.crop_h is not None:
                    h_start, h_end = self.crop_h
                    rows = min(rows, h_end) - max(0, h_start)
                if self.crop_w is not None:
                    w_start, w_end = self.crop_w
                    cols = min(cols, w_end) - max(0, w_start)
                row_patches = ((rows - self.patch_size[0]) // self.patch_stride[0]) + 1
                col_patches = ((cols - self.patch_size[1]) // self.patch_stride[1]) + 1
                patches.append(row_patches * col_patches)
            patches = np.cumsum(np.array(patches))
        self.patches = patches

    def __len__(self) -> int:
        """Return the total number of patches across all files."""
        if self.patches is not None:
            return self.patches[-1]
        else:
            return len(self.files)

    def __getitem__(self, index: int) -> np.ndarray:
        file_index = index
        patch_index = None
        if self.patches is not None:
            file_index = np.searchsorted(self.patches, index, side="right")
            if file_index == 0:
                patch_index = index
            else:
                patch_index = index - self.patches[file_index - 1]
        p = self.files[file_index]

        data = np.load(p.as_posix())

        if self.crop_h is not None:
            h_start, h_end = self.crop_h
            data = data[max(0, h_start) : min(data.shape[0], h_end), :]
        if self.crop_w is not None:
            w_start, w_end = self.crop_w
            data = data[:, max(0, w_start) : min(data.shape[1], w_end)]

        if patch_index is not None:
            rows, cols = p.stem.split("_")
            rows, cols = int(rows), int(cols)
            col_patches = ((cols - self.patch_size[1]) // self.patch_stride[1]) + 1

            row_idx = patch_index // col_patches
            col_idx = patch_index % col_patches
            data = data[
                row_idx * self.patch_stride[0] : row_idx * self.patch_stride[0]
                + self.patch_size[0],
                col_idx * self.patch_stride[1] : col_idx * self.patch_stride[1]
                + self.patch_size[1],
            ]

        if self.norm is not None:
            stats_file = os.path.join(p.parent, ".stats.json")
            if not os.path.exists(stats_file):
                raise FileNotFoundError(
                    f"Stats file not found at {stats_file}. Required for normalization."
                )
            with open(os.path.join(p.parent, ".stats.json"), "r") as f:
                stats = json.load(f)
            minus = stats["mean"] if self.norm == "z-score" else stats["min"]
            denominator = (
                stats["std"] if self.norm == "z-score" else stats["max"] - stats["min"]
            )
            data = (data - minus) / denominator
        return data

    def __str__(self) -> str:
        return f"NumpyPatchReader at '{self.root_dir}' ({len(self.files)} files and {self.patches[-1] if self.patches is not None else len(self.files)} patches)"
