from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import functools
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Literal, TypedDict, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision.transforms import v2 as tv2
from lightning.pytorch import LightningDataModule
import tqdm

from brain_image.configs import BaseConfig, GlobalConfig
import multiprocessing as mp


class DataConfig(BaseConfig, ABC):
    data_path: Path

    batch_size: int = 128
    val_batch_size: int | None = None
    test_batch_size: int | None = None

    limit_train_size: float = 1.0
    limit_val_size: float = 1.0
    limit_test_size: float = 1.0

    num_workers: int | None = None

    def create_datamodule(self) -> DataModule:
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


class EEGDatasetConfig(DataConfig):
    data_path: Path = GlobalConfig.DATA_DIR / "things-eeg2"

    imgs_dir: str = "imgs"
    eeg_dir: str = "prepared"

    train_imgs_per_concept: int = 10
    test_imgs_per_concept: int = 1
    subs: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    preload_cache: bool = True

    seed: int = 42


class TensorCache:
    def __init__(
        self,
        cache_path: Path = Path("cache/tensorcache"),
        memory_cache_size: int = 512000,
        use_encrypt: bool = False,
        overwrite: bool = True,
    ):
        self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.memory_cache_size = memory_cache_size
        self.memory_cache = {}
        self.memory_cache_keys = []
        self.use_encrypt = use_encrypt
        self.overwrite = overwrite

    def _get_tensor_path(self, *keys: str) -> Path:
        encoded_path = self._encode_keys(*keys, use_encrypt=self.use_encrypt)
        full_path = self.cache_path / encoded_path
        return full_path.with_suffix(".pt")

    def save(self, tensor: torch.Tensor, *keys: str):
        path = self._get_tensor_path(*keys)
        self._add_to_memory_cache(path, tensor)

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not self.overwrite:
            raise FileExistsError(f"File already exists: {path}")

        torch.save(tensor, path)

    def get(self, *keys: str) -> torch.Tensor:
        path = self._get_tensor_path(*keys)
        if path in self.memory_cache:
            return self.memory_cache[path]

        tensor = torch.load(path)

        self._add_to_memory_cache(path, tensor)
        return tensor

    def _add_to_memory_cache(self, path: Path, tensor: torch.Tensor):
        self.memory_cache_keys.append(path)
        self.memory_cache[path] = tensor

        if len(self.memory_cache_keys) > self.memory_cache_size:
            oldest_key = self.memory_cache_keys.pop(0)
            self.memory_cache.pop(oldest_key)

    @staticmethod
    def _encode_keys(*keys: str, use_encrypt: bool = False) -> str:
        if use_encrypt:
            h = hashlib.sha1()
            for key in keys:
                h.update(key.encode())
            return h.hexdigest()
        else:
            return "/".join(keys)


class DataModule(LightningDataModule, ABC):
    def __init__(self, config: DataConfig):
        super().__init__()

        self.config = config
        self.num_batches: dict[str, int | None] = {"train": None, "val": None, "test": None}

        self.datasets: dict[str, Dataset] = {split: self._create_dataset(split) for split in ["train", "val", "test"]}
        self.dataloaders: dict[str, torch.utils.data.DataLoader | None] = {}

    @abstractmethod
    def get_metadata(self) -> dict:
        raise NotImplementedError

    def get_dataloader(self, split: Literal["train", "val", "test"]):
        if (dataloader := self.dataloaders.get(split)) is not None:
            return dataloader

        dataloader = self._create_dataloader(split)
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

    def _get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset:
        if (dataset := self.datasets.get(split)) is not None:
            return dataset

        dataset = self._create_dataset(split)
        self.datasets[split] = dataset
        return dataset

    @abstractmethod
    def _create_dataset(self, split) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def _create_dataloader(self, split) -> torch.utils.data.DataLoader:
        raise NotImplementedError



class EmbeddingsMap(TypedDict):
    align_img_latent: str | None
    prior_img_latent: str | None
    recon_latent: str | None


class SampleType(TypedDict):
    img_path: str
    idx: int
    sub: int
    eeg_data: torch.Tensor
    align_img_latent: torch.Tensor | None
    prior_img_latent: torch.Tensor | None
    recon_img_latent: torch.Tensor | None


class EEGDataModule(DataModule):
    def __init__(
        self,
        config: EEGDatasetConfig,
        tensor_cache: TensorCache,
        embeddings_map: EmbeddingsMap,
    ):
        self.config: EEGDatasetConfig = config
        self.rng = np.random.default_rng(config.seed)
        self.tensor_cache = tensor_cache
        self.embeddings_map = embeddings_map
        super().__init__(config)

    def get_metadata(self) -> dict:
        return {}

    def _create_dataset(
        self, split: Literal["train", "val", "test"]
    ) -> EEGDataset:
        return EEGDataset(
            self.config,
            split=split,
            tensor_cache=self.tensor_cache,
            embeddings_map=self.embeddings_map,
            limit_size=self.config.get_limit_size(split),
            preload_cache=self.config.preload_cache
        )

    def _create_dataloader(
        self,
        split: Literal["train", "val", "test"],
        **kwargs,
    ) -> torch.utils.data.DataLoader:

        match split:
            case "train":
                shuffle = True
                
            case "val":
                shuffle = False

            case "test":
                shuffle = False 

        num_workers = self.config.num_workers or mp.cpu_count()
        dataloader_args = {
            "batch_size": self.config.get_batch_size(split),
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        dataloader_args.update(kwargs)

        dataset = self.datasets[split]
        return torch.utils.data.DataLoader(
            dataset,
            **dataloader_args,
        )


class EEGDataset(Dataset):
    def __init__(
        self,
        config: EEGDatasetConfig,
        split: Literal["train", "val", "test"],
        tensor_cache: TensorCache,
        embeddings_map: EmbeddingsMap = cast(EmbeddingsMap, {}),
        standardize_embeddings: list[str] = ["prior_img_latent"],
        limit_size: float = 1.0,
        preload_cache: bool = True,
    ):
        self.config = config
        self.split: Literal["train", "val", "test"] = split
        self.tensor_cache = tensor_cache
        self.embeddings_map = embeddings_map
        self.standardize_embeddings = standardize_embeddings
        self.limit_size = limit_size
 
        logging.info(f"Loading {split} dataset")
        prepared_data: list[dict] = []
        split_dir = "train" if split == "train" else "test"
        for sub in self.config.subs:
            prepared_data.extend(
                torch.load(
                    self.config.data_path
                    / self.config.eeg_dir
                    / f"sub-{sub:02}"
                    / f"{split_dir}.pt"
                )
            )
        self.prepared_data = prepared_data
        logging.info(f"Loaded {len(self.prepared_data)} samples")

        if self.limit_size < 1.0:
            logging.info(f"Limiting dataset size to {self.limit_size * 100:.1f}%")
            idxs = np.random.choice(
                len(self.prepared_data),
                int(len(self.prepared_data) * self.limit_size),
                replace=False,
            )
            self.prepared_data = [self.prepared_data[i] for i in idxs]

        if preload_cache:
            self._preload_cache()


    def __len__(self):
        return len(self.prepared_data)

    def __getitem__(self, idx: int):
        item = self.prepared_data[idx]

        sample = {
            "img_path": str(item["img_path"]),
            "eeg_data": item["eeg"],
            "idx": item["idx"],
            "sub": item["sub"],
            **self._get_embeddings(item["img_path"])
        }

        return sample

    def _get_embeddings(self, img_path: Path):
        return {
            key: self._get_image_latent_from_cache(img_path, str(value), self.split)
            for key, value in self.embeddings_map.items()
            if value is not None
        }

    def _get_image_latent_from_cache(
        self, img_path: Path, model_name: str, split: Literal["train", "val", "test"]
    ) -> torch.Tensor:
        split = "train" if split == "train" else "test"
        return self.tensor_cache.get(str(img_path), model_name, split)

    def _preload_cache(self, parallel: bool = True):
        if parallel:
            with ThreadPoolExecutor() as executor:
                logging.info(
                    f"Preloading latents in parallel with {executor._max_workers} workers"
                )
                outs = executor.map(self.__getitem__, range(len(self)))
                num_items = sum(1 for _ in outs)
                logging.info(f"Preloaded {num_items} latents")
        else:
            for i in tqdm.tqdm(range(len(self)), desc="Preloading latents"):
                self.__getitem__(i)


def load_image_from_path(path: Path, mode: str | None = None) -> Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    match mode:
        case "rgb":
            mode_value = torchvision.io.ImageReadMode.RGB
        case None:
            mode_value = torchvision.io.ImageReadMode.UNCHANGED
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    img = torchvision.io.decode_image(str(path), mode=mode_value)
    return img


def batch_load_images(
    paths: Iterable[Path],
    parallel: bool = False,
    progressbar: bool = False,
    mode: str | None = None,
) -> Tensor:
    if parallel:
        with mp.Pool() as pool:
            imgs = pool.map(functools.partial(load_image_from_path, mode=mode), paths)
    else:
        imgs = [
            load_image_from_path(path, mode=mode)
            for path in tqdm.tqdm(
                list(paths), disable=not progressbar, desc="Loading images"
            )
        ]

    imgs = torch.stack(imgs, dim=0)
    return imgs


def load_eeg_data(
    eeg_path: Path,
) -> tuple[Tensor, Tensor, Tensor, list[str]]:
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG data not found: {eeg_path}")

    # Load the EEG data
    eeg_pickle = np.load(eeg_path, allow_pickle=True)
    raw_eeg = eeg_pickle["preprocessed_eeg_data"]
    channel_names = eeg_pickle["ch_names"]
    times = eeg_pickle["times"]

    raw_eeg = torch.from_numpy(raw_eeg).float()
    times = torch.from_numpy(times).float()
    idxs = torch.arange(len(raw_eeg))

    return raw_eeg, idxs, times, channel_names


def preprocess_eeg_data(
    eeg_data: Tensor,
    idxs: Tensor,
    interpolate_size: tuple[int, int] | None = None,
    normalize: bool = True,
    unpack_repetitions: bool = True,
) -> tuple[Tensor, Tensor]:
    """Preprocess the EEG data by averaging over the number of repetitions.

    Args:
        eeg_data (numpy.ndarray): The EEG data to preprocess. <concepts, repetitions, channels, timesteps>

    Returns:
        numpy.ndarray: The preprocessed EEG data. <concepts, channels, timesteps>
    """
    # Average over the number of repetitions
    if interpolate_size is not None:
        preprocessed_data = torch.nn.functional.interpolate(
            eeg_data, size=interpolate_size, mode="nearest"
        )
    else:
        preprocessed_data = eeg_data

    if unpack_repetitions:
        num_repetitions = preprocessed_data.size(1)
        idxs = idxs.repeat_interleave(num_repetitions, dim=0)
        preprocessed_data = preprocessed_data.reshape(-1, *preprocessed_data.shape[2:])
    else:
        preprocessed_data = torch.mean(preprocessed_data, dim=1)

    if normalize:
        preprocessed_data = (
            preprocessed_data - preprocessed_data.mean(dim=0, keepdim=True)
        ) / (np.sqrt(2) * preprocessed_data.std(dim=0, keepdim=True))

    return preprocessed_data, idxs


def get_image_paths(
    image_dir: Path,
    split: Literal["train", "test"],
    extensions: tuple[str, ...] = (".jpg", ".png", ".jpeg"),
) -> list[Path]:
    """Get all image paths from a directory."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    if split == "train":
        image_dir = image_dir / "training_images"

    elif split == "test":
        image_dir = image_dir / "test_images"

    img_paths = [
        img_path
        for concept_dir in sorted(image_dir.iterdir())
        for img_path in sorted(concept_dir.iterdir())
        if img_path.is_file() and img_path.suffix in extensions
    ]

    return img_paths


def load_all_eeg_data(
    eeg_paths: list[Path], preprocess_configs: dict | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    preprocess_configs = preprocess_configs or {}

    all_eeg_data = []
    all_idxs = []
    all_times = None
    all_ch_names = []

    for eeg_path in eeg_paths:
        eeg_data, idxs, times, ch_names = load_eeg_data(eeg_path)
        eeg_data, idxs = preprocess_eeg_data(eeg_data, idxs, **preprocess_configs)

        all_eeg_data.append(eeg_data)

        if all_times is None:
            all_times = times

        if not all_ch_names:
            all_ch_names = ch_names

        all_idxs.append(idxs)

    if all_times is None:
        all_times = torch.tensor([])

    return torch.stack(all_eeg_data), torch.stack(all_idxs), all_times, all_ch_names
