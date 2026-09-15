import numpy as np
from minerva.transforms.transform import Padding

class SafePadding(Padding):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        pad_h = max(0, self.target_h_size - h)
        pad_w = max(0, self.target_w_size - w)

        pad_kwargs = {"mode": self.padding_mode}
        if self.padding_mode == "constant":
            pad_kwargs["constant_values"] = (
                self.mask_padding_value
                if np.issubdtype(x.dtype, np.unsignedinteger)
                else self.padding_value
            )
            
        if len(x.shape) == 2:
            padded = np.pad(x, ((0, pad_h), (0, pad_w)), **pad_kwargs)
            padded = np.expand_dims(padded, axis=2) 
        else:
            padded = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), **pad_kwargs)

        return padded
