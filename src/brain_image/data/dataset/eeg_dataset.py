from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging

import torch
import tqdm
from brain_image.configs import BaseConfig
from torch.utils.data import Dataset
from typing import Literal, Sequence, cast


from abc import ABC, abstractmethod
from pathlib import Path

from brain_image.data.data import DSPLIT, EEGSampleT, LatentGroupT, LatentStats, LatentTypeMapT, get_embeddings_stats
from brain_image.data.tensorcache import TensorCache
from brain_image.model.encoder.encoder import EncoderName, LatentName


class DataConfig(BaseConfig, ABC):
    data_path: Path

    batch_size: int = 128
    val_batch_size: int | None = None
    test_batch_size: int | None = None

    limit_train_size: float = 1.0
    limit_val_size: float = 1.0
    limit_test_size: float = 1.0

    num_workers: int | None = None

    @abstractmethod
    def create_dataset(self, split: Literal["train", "val", "test"], *args, **kwargs) -> Dataset:
        raise NotImplementedError


    def get_shuffle(self, split: Literal["train", "val", "test"]):
        match split:
            case "train":
                return True
            case "val":
                return False
            case "test":
                return False

    def get_limit_size(self, split: Literal["train", "val", "test"]):
        match split:
            case "train":
                return self.limit_train_size
            case "val":
                return self.limit_val_size
            case "test":
                return self.limit_test_size

    def get_batch_size(self, split: Literal["train", "val", "test"]):
        match split:
            case "train":
                return self.batch_size
            case "val":
                return self.val_batch_size or self.batch_size
            case "test":
                return self.test_batch_size or self.batch_size


class EEGDataset(Dataset, ABC):
    def __init__(
        self,
        config: EEGDatasetConfig,
        split: Literal["train", "val", "test"],
        tensor_cache: TensorCache | None = None,
        embeddings_map: LatentTypeMapT | None = None,
        embeddings_to_compute_stats: Sequence[str] = ("prior_img_latent",),
        limit_size: float | None = None,
        limit_shuffle: bool = True,
        preload_cache: bool | None = None,
        compute_stats: bool | None = None,
    ):
        tensor_cache = tensor_cache or TensorCache()
        embeddings_map = embeddings_map or {
            "align_img_latent": None,
            "prior_img_latent": None,
            "low_level_latent": None,
            "eeg_latent": None,
        }

        logging.info(f"Setting up latents loader with embeddings map:")
        for k, v in embeddings_map.items():
            logging.info(f"  {k}: {v}")

        compute_stats = split == "train" if compute_stats is None else compute_stats
        limit_size = config.get_limit_size(split) if limit_size is None else limit_size
        preload_cache = config.preload_cache if preload_cache is None else preload_cache

        self.config = config
        self.split: Literal["train", "val", "test"] = split
        self.tensor_cache = tensor_cache
        self.embeddings_map = embeddings_map
        self.embeddings_to_compute_stats = embeddings_to_compute_stats
        self.limit_size = limit_size
        self.compute_stats = compute_stats
        self.embedding_stats: dict[LatentName, LatentStats] = {}
        self.eeg_stats = {}

        logging.info(f"Loading EEG")
        self.eeg = torch.stack(
            [self.load_eeg_from_path(eeg_path) for eeg_path in self.get_eeg_paths()]
        ).float()  # <sub, image, channel, time>

        logging.info(f"Reducing dataset size to {limit_size * 100:.2f}%")
        self.limit_data_size(limit_size, limit_shuffle)
        logging.info(f"Reduced dataset size to: {len(self)}")

        logging.info(f"Preparing {split} dataset...")
        self.prepare()
        logging.info(f"Prepared dataset of size: {len(self)}")

        if preload_cache:
            self._preload_cache()

        if compute_stats:
            self.embedding_stats = self._compute_embedding_stats()

    @abstractmethod
    def load_eeg_from_path(self, eeg_path: Path) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_eeg_paths(
        self, split: DSPLIT | None = None, subs: list[int] | None = None
    ) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def get_image_paths(self) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> EEGSampleT:
        raise NotImplementedError

    @abstractmethod
    def prepare(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def limit_data_size(self, limit_size: float, limit_shuffle: bool = True) -> None:
        raise NotImplementedError

    def get_embeddings(self, img_path: Path) -> LatentGroupT:
        return cast(
            LatentGroupT,
            {
                key: self._get_image_latent_from_cache(
                    img_path, cast(EncoderName, value), self.split
                )
                for key, value in self.embeddings_map.items()
                if value is not None
            },
        )

    def _compute_embedding_stats(self) -> dict[LatentName, LatentStats]:
        img_paths = self.get_image_paths()

        embedding_types = [
            str(v)
            for k, v in self.embeddings_map.items()
            if v in self.embeddings_to_compute_stats and v is not None
        ]
        embedding_stats = get_embeddings_stats(
            tensorcache=self.tensor_cache,
            img_paths=img_paths,
            embedding_names=embedding_types,  # type: ignore
            split="train",
        )

        # Convert from specific encoder name to general encoding role
        # E.g. ATMS -> EEG_encoder
        mapped_stats = {
            k: embedding_stats[v]
            for k, v in self.embeddings_map.items()
            if v in embedding_stats
        }
        mapped_stats = cast(dict[LatentName, LatentStats], mapped_stats)
        return mapped_stats

    def _get_image_latent_from_cache(
        self,
        img_path: Path,
        model_name: EncoderName,
        split: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        return self.tensor_cache.get_latent(img_path, model_name, split)

    def _preload_cache(self, parallel: bool = True):
        if parallel:
            num_workers = self.config.num_workers or None
            with ThreadPoolExecutor(num_workers) as executor:
                logging.info(
                    f"Preloading latents in parallel with {executor._max_workers if num_workers is None else num_workers} workers"
                )
                outs = executor.map(self.__getitem__, range(len(self)))
                num_items = sum(1 for _ in outs)
                logging.info(f"Preloaded {num_items} latents")
        else:
            for i in tqdm.tqdm(range(len(self)), desc="Preloading latents"):
                self.__getitem__(i)

    def get_embedding_stats(self) -> dict[LatentName, LatentStats]:
        return self.embedding_stats


class EEGDatasetConfig(DataConfig, ABC):
    data_path: Path
    dataset: str
    subs: list[int] | None
    num_channels: int
    time_length: int

    preload_cache: bool = True

    @abstractmethod
    def create_dataset(self, split, *args, **kwargs) -> EEGDataset:
        raise NotImplementedError