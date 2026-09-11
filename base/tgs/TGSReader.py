from torchvision.datasets import ImageFolder
import numpy as np
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from scipy.io import loadmat
from .utils import reduce_taxonomic_diversity
import time

class TGSReader:
    def __init__(self, root: str, split='train'):
        self.root = Path(root)
        self.split = split
        self.data_dir = self.root / self.split / 'data'

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        
        self.file_paths = sorted(list(self.data_dir.glob("*.npy")))
        print(f"[{self.split.upper()}] Loaded {len(self.file_paths)} .npy files from {self.data_dir}")

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        last_error = None

        for attempt in range(10):
            try:
                data = np.load(file_path, allow_pickle=True)

                if isinstance(data, np.ndarray):
                    # Saved as [image, mask]
                    img, mask = data[0], data[1]
                
                else:
                    # Saved as a multi-channel array, e.g., shape (2, H, W)
                    if data.ndim == 3 and data.shape[0] >= 2:
                        img, mask = data[0], data[1]
                    else:
                        img = data
                        mask = np.zeros_like(img)
                
                img = torch.from_numpy(img).float()
                mask = torch.from_numpy(mask).float()

                if img.ndim == 2:
                    img = img.unsqueeze(0)  # Shape becomes [1, H, W]
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)  # Shape becomes [3, H, W]

                # Ensure mask is [1, H, W] for binary segmentation
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                
                # Normalize mask to strictly 0 and 1 (if not already)
                mask = (mask > 0.5).float()

                return img, msk
            
            except (OSError, ValueError, EOFError) as e:
                last_error = e
                time.sleep(0.05 * (attempt + 1))
        
        raise RuntimeError(f"Failed to read {file_path} after 10 attempts") from last_error
