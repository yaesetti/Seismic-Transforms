from deeprock.datamodules.base import DeepRockDataModule
from deeprock.datamodules.channel_identification import (
    BinaryChannelIdentificationDataModule,
    ChannelIdentificationDataModule,
)
from deeprock.datamodules.facies import FaciesDataModule
from deeprock.datamodules.faults import FaultsDataModule
from deeprock.datamodules.salt import SaltDataModule
from deeprock.datamodules.seismic_attribute_calculation import (
    SeismicAttributeDataModule,
)
from deeprock.datamodules.super_resolution import SuperResolutionDataModule

__all__ = [
    "DeepRockDataModule",
    "BinaryChannelIdentificationDataModule",
    "ChannelIdentificationDataModule",
    "FaciesDataModule",
    "FaultsDataModule",
    "SaltDataModule",
    "SeismicAttributeDataModule",
    "SuperResolutionDataModule",
]
