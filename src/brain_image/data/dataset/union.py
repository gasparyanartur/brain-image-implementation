from typing import Literal, cast

from brain_image.data.dataset.eeg_dataset import EEGDataset

from brain_image.data.dataset.dummy_eeg_dataset import DummyEEGDataset

from brain_image.data.dataset.alljoined_eeg2_dataset import (
    AlljoinedEEG2DatasetConfig,
    AlljoinedEEG2Dataset,
)
from brain_image.data.dataset.eeg_dataset import EEGDatasetConfig
from brain_image.data.dataset.things_eeg2_dataset import (
    ThingsEEG2DatasetConfig,
    ThingsEEG2Dataset,
)
from brain_image.data.dataset.dummy_eeg_dataset import DummyEEGDatasetConfig, DummyEEGDataset


EEGDatasetKey = Literal["alljoined-eeg2", "things-eeg2", "dummy"]
EEGDatasetConfigType = (
    AlljoinedEEG2DatasetConfig | ThingsEEG2DatasetConfig | DummyEEGDatasetConfig
)


def create_dataset(config, *args, **kwargs) -> EEGDataset:
    match config.dataset:
        case "things-eeg2":
            return ThingsEEG2Dataset(config, *args, **kwargs)  # type: ignore
        case "alljoined-eeg2":
            return AlljoinedEEG2Dataset(config, *args, **kwargs)  # type: ignore
        case "dummy":
            return DummyEEGDataset(config, *args, **kwargs)
        case _:
            raise ValueError(f"Unrecognized dataset type: {config.dataset}")


def resolve_dataset_config(config: dict | EEGDatasetConfig) -> EEGDatasetConfig:
    if isinstance(config, EEGDatasetConfig):
        return config

    match config["dataset"]:
        case "alljoined-eeg2":
            return AlljoinedEEG2DatasetConfig(**config)
        case "things-eeg2":

            return ThingsEEG2DatasetConfig(**config)
        case "dummy":
            return DummyEEGDatasetConfig(**config)
        case _:
            raise ValueError(f"Unknown dataset: {config['dataset']}")
