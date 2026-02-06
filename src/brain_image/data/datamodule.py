from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from brain_image.configs import get_device_str


import torch


import logging
import multiprocessing as mp
from typing import Literal, cast

from brain_image.data.data import DSPLIT
from brain_image.data.dataset.eeg_dataset import DataConfig, EEGDataset, EEGDatasetConfig
from brain_image.data.tensorcache import TensorCache
from brain_image.model.encoder.encoder import EncoderName
from brain_image.model.encoder.img_encoder.union import ImageEncoderName


class DataModule(LightningDataModule, ABC):
    def __init__(self, config: DataConfig):
        super().__init__()

        self.config = config
        self.num_batches: dict[str, int | None] = {
            "train": None,
            "val": None,
            "test": None,
        }

        self.datasets: dict[str, Dataset | None] = {split: None for split in ["train", "val", "test"]}
        self.dataloaders: dict[str, torch.utils.data.DataLoader | None] = {}

    @property
    @abstractmethod
    def get_embedding_stats(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict:
        raise NotImplementedError

    def get_dataloader(self, split: Literal["train", "val", "test"]):
        if (dataloader := self.dataloaders.get(split)) is not None:
            return dataloader

        dataloader = self.create_dataloader(split)
        self.dataloaders[split] = dataloader
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def get_num_batches(self, split: Literal["train", "val", "test"]) -> int:
        if (num_batches := self.num_batches.get(split)) is not None:
            return num_batches

        dataloader = self.get_dataloader(split)
        num_batches = len(dataloader)
        self.num_batches[split] = num_batches

        return num_batches

    def get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset:
        if (dataset := self.datasets.get(split)) is not None:
            return dataset

        dataset = self.create_dataset(split)
        self.datasets[split] = dataset
        return dataset

    @abstractmethod
    def create_dataset(self, split) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def create_dataloader(self, split) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    def cleanup(self) -> None:
        dataloader_keys = list(self.dataloaders.keys())
        for key in dataloader_keys:
            if self.dataloaders[key] is None:
                continue
            del self.dataloaders[key]


class EEGDataModule(DataModule):
    def __init__(
        self,
        config: EEGDatasetConfig,
        tensor_cache: TensorCache | None = None,
        embeddings_key_to_name: dict[str, EncoderName | None] | None = None,
        load_embedding_stats: list[ImageEncoderName] | None = None,
    ):
        super().__init__(config)

        tensor_cache = tensor_cache or TensorCache()
        embeddings_key_to_name = embeddings_key_to_name or {}
        embeddings_key_to_name = {k: v for k, v in embeddings_key_to_name.items() if v is not None}

        load_embedding_stats = load_embedding_stats or []
        self.config: EEGDatasetConfig = config
        self.tensor_cache = tensor_cache
        self.embeddings_key_to_name = embeddings_key_to_name
        self.load_embedding_stats = load_embedding_stats

        logging.info(f"Got embedding stats for: {self.get_embedding_stats().keys()}")

    def get_embedding_stats(self, split: DSPLIT = "train") -> dict:
        return self.get_dataset(split).get_embedding_stats()

    def get_metadata(self) -> dict:
        return {}

    def create_dataset(self, split: Literal["train", "val", "test"], *args, **kwargs) -> EEGDataset:
        self.config
        return self.config.create_dataset(
            split,
            tensor_cache=self.tensor_cache,
            embeddings_key_to_name=self.embeddings_key_to_name,
            load_embedding_stats=self.load_embedding_stats,
        )

    def create_dataloader(
        self,
        split: Literal["train", "val", "test"],
        **kwargs,
    ) -> torch.utils.data.DataLoader:

        shuffle = split == "train"
        num_workers = self.config.num_workers or min(32, mp.cpu_count())
        device = get_device_str()
        dataloader_args = {
            "batch_size": self.config.get_batch_size(split),
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": device != "cpu",
            "persistent_workers": (split == "train") and (num_workers > 0),
            "drop_last": False,
        }
        dataloader_args.update(kwargs)

        dataset = self.get_dataset(split)
        return torch.utils.data.DataLoader(
            dataset,
            **dataloader_args,
        )

    def get_dataset(self, split: Literal["train", "val", "test"]) -> EEGDataset:
        return cast(EEGDataset, super().get_dataset(split))

    def get_embeddings_stats(self):
        return self.get_dataset("train").get_embedding_stats()
