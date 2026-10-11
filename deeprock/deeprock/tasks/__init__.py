from deeprock.tasks.base import DeepRockTask
from deeprock.tasks.channel_identification import (
    ChannelIdentificationTask,
    BinaryChannelIdentificationTask,
)
from deeprock.tasks.facies import FaciesTask
from deeprock.tasks.faults import FaultsTask
from deeprock.tasks.salt import SaltTask
from deeprock.tasks.seismic_attribute_calculation import (
    SeismicAttributeTask,
)
from deeprock.tasks.super_resolution import SuperResolutionTask

__all__ = [
    "DeepRockTask",
    "BinaryChannelIdentificationTask",
    "ChannelIdentificationTask",
    "FaciesTask",
    "FaultsTask",
    "SaltTask",
    "SeismicAttributeTask",
    "SuperResolutionTask",
]
