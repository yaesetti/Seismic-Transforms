from deeprock.metrics import BinaryTGSMeanIoU
from deeprock.tasks.base import BinaryTask
from deeprock.datamodules.salt import SaltDataModule

from pathlib import Path
import json
from typing import List, Union

import numpy as np
from torchmetrics.classification import (
    # binary
    BinaryF1Score,
    BinaryJaccardIndex,
)
import lightning as L

from deeprock.model import DeepRockModel


class SaltTask(BinaryTask):
    def __init__(self, **datamodule_kwargs):
        super().__init__(SaltDataModule, **datamodule_kwargs)