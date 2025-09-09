from __future__ import annotations

from functools import lru_cache
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Literal, TypedDict, cast

import numpy as np
import torch
from torch import Tensor, manual_seed
from torch.utils.data import Dataset, random_split, Subset
import torchvision
from torchvision.transforms import v2 as tv2
from lightning.pytorch import LightningDataModule

from brain_image.configs import DEFAULT_BATCH_SIZE, BaseConfig, GlobalConfig


def encode_keys(*keys: str, use_encrypt: bool = False) -> str:
    if use_encrypt:
        h = hashlib.sha1()
        for key in keys:
            h.update(key.encode())
        return h.hexdigest()
    else:
        return "/".join(keys)


class TensorCache:
    def __init__(
        self,
        cache_path: Path = Path("cache/tensorcache"),
        memory_cache_size: int = 65536,
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
        encoded_path = encode_keys(*keys, use_encrypt=self.use_encrypt)
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


class DataConfig(BaseConfig, ABC):
    data_path: Path
    batch_size: int = DEFAULT_BATCH_SIZE
    val_batch_size: int = DEFAULT_BATCH_SIZE
    eval_batch_size: int = DEFAULT_BATCH_SIZE
    shuffle_train: bool = True
    limit_train_size: float = 1.0
    limit_val_size: float = 1.0
    limit_test_size: float = 1.0
    num_workers: int = 32

    def create_datamodule(self) -> DataModule:
        raise NotImplementedError


class DataModule(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()

        self.config = config

    @abstractmethod
    def get_metadata(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_train_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_val_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_test_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def train_dataloader(self):
        return self._create_dataloader(self.get_train_dataset())

    def val_dataloader(self):
        return self._create_dataloader(self.get_val_dataset(), shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.get_test_dataset(), shuffle=False)

    def _create_dataloader(self, dataset, shuffle=True, batch_size=None):
        if batch_size is None:
            batch_size = self.config.batch_size
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )


class EEGDatasetConfig(DataConfig):
    data_path: Path = GlobalConfig.DATA_DIR / "things-eeg2"

    imgs_dir: str = "imgs"
    eeg_dir: str = "eeg"

    train_imgs_per_concept: int = 10
    test_imgs_per_concept: int = 1
    subs: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def get_subset_dataset(
    rng: np.random.Generator, dataset: Dataset, subset_frac: float
) -> Dataset:
    subset_size = int(len(dataset) * subset_frac)
    indices = list(range(len(dataset)))
    subset_indices = rng.choice(indices, subset_size, replace=False)
    return Subset(dataset, subset_indices.tolist())


class EEGDataModule(DataModule):
    def __init__(self, config: EEGDatasetConfig):
        super().__init__(config)
        self.config: EEGDatasetConfig = config
        self.rng = np.random.default_rng(42)

    def get_metadata(self) -> dict:
        return {}

    def get_train_dataset(self) -> EEGDataset:
        dataset = EEGDataset(
            self.config,
            split="train",
        )

        # Apply train size limit if specified
        if self.config.limit_train_size < 1.0:
            dataset = get_subset_dataset(
                self.rng, dataset, self.config.limit_train_size
            )
            logging.info(
                f"Limited train dataset to {len(dataset)} samples ({self.config.limit_train_size * 100:.1f}%)"
            )

        return dataset

    def get_val_dataset(self) -> EEGDataset:
        dataset = EEGDataset(self.config, split="test")

        # Apply validation size limit if specified
        if self.config.limit_val_size < 1.0:
            dataset = get_subset_dataset(self.rng, dataset, self.config.limit_val_size)
            logging.info(
                f"Limited validation dataset to {len(dataset)} samples ({self.config.limit_val_size * 100:.1f}%)"
            )

        return dataset

    def get_test_dataset(self) -> EEGDataset:
        dataset = EEGDataset(self.config, split="test")

        # Apply test size limit if specified
        if self.config.limit_test_size < 1.0:
            dataset = get_subset_dataset(self.rng, dataset, self.config.limit_test_size)
            logging.info(
                f"Limited test dataset to {len(dataset)} samples ({self.config.limit_test_size * 100:.1f}%)"
            )

        return dataset

    def train_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        return self._create_dataloader(
            self.get_train_dataset(),
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            **kwargs
        )

    def val_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        return self._create_dataloader(
            self.get_val_dataset(), batch_size=self.config.val_batch_size, shuffle=False, **kwargs
        )

    def test_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        return self._create_dataloader(
            self.get_test_dataset(),
            batch_size=self.config.val_batch_size,
            shuffle=False,
            **kwargs
        )

    def _create_dataloader(
        self, dataset: EEGDataset, shuffle: bool = True, batch_size: int | None = None, **kwargs
    ) -> torch.utils.data.DataLoader:
        if batch_size is None:
            batch_size = self.config.batch_size

        dataloader_args = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": self.config.num_workers,
            "pin_memory": True,
            "persistent_workers": self.config.num_workers > 0,
        }
        dataloader_args.update(kwargs)

        return torch.utils.data.DataLoader(
            dataset,
            **dataloader_args,
        )


class EEGDataset(Dataset):
    def __init__(
        self,
        config: EEGDatasetConfig,
        split: Literal["train", "test"],
    ):
        self.config = config
        self.split = split
        
        self.prepared_data = torch.load(self.config.data_path / "prepared" / f"{split}.pt")
        self.prepared_data = [
            sample for sample in self.prepared_data if sample["sub"] in self.config.subs
        ]

    def __len__(self):
        return len(self.prepared_data)

    def __getitem__(self, idx: int):
        item = self.prepared_data[idx]

        return {
            "img_path": str(item["img_path"]),
            "eeg_data": item["eeg"],
            "idx": idx,
        }


def prepare_datasets(
    config: EEGDatasetConfig,
    seed: int = 42,
    train_val_split: float = 0.8,
    use_test_as_val: bool = True,
) -> tuple[EEGDataset, EEGDataset, EEGDataset]:
    train_dataset = EEGDataset(config, split="train")
    test_dataset = EEGDataset(config, split="test")

    if use_test_as_val:
        val_dataset = EEGDataset(config, split="test")

    else:
        split_rng = manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_dataset, [train_val_split, 1 - train_val_split], generator=split_rng
        )

    train_dataset = cast(EEGDataset, train_dataset)
    val_dataset = cast(EEGDataset, val_dataset)
    test_dataset = cast(EEGDataset, test_dataset)

    return train_dataset, val_dataset, test_dataset


def load_image_from_path(path: Path) -> Tensor:
    """Load an image from a given path."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = torchvision.io.decode_image(str(path))
    return img


def batch_load_images(paths: Iterable[Path]) -> Tensor:
    """Load a batch of images from a list of paths."""
    imgs = [load_image_from_path(path) for path in paths]
    imgs = torch.stack(imgs, dim=0)
    return imgs


def load_eeg_data(
    eeg_path: Path,
) -> tuple[Tensor, Tensor, list[str]]:
    """Load EEG data from a given path."""
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG data not found: {eeg_path}")

    # Load the EEG data
    eeg_pickle = np.load(eeg_path, allow_pickle=True).item()
    raw_eeg = eeg_pickle["preprocessed_eeg_data"]
    channel_names = eeg_pickle["ch_names"]
    times = eeg_pickle["times"]

    raw_eeg = torch.from_numpy(raw_eeg).float()
    times = torch.from_numpy(times).float()

    return raw_eeg, times, channel_names


def preprocess_image(
    image: torch.Tensor, img_size: list[int] = [224, 224]
) -> torch.Tensor:
    image = tv2.functional.resize(
        image, list(img_size), interpolation=tv2.InterpolationMode.BICUBIC
    )
    image = image / 255.0
    return image


def preprocess_eeg_data(
    eeg_data: Tensor,
    interpolate_size: tuple[int, int] | None = None,
) -> Tensor:
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

    preprocessed_data = torch.mean(preprocessed_data, dim=1)

    return preprocessed_data


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
    eeg_paths: list[Path],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    all_eeg_data = []
    all_times = None
    all_ch_names = []

    for eeg_path in eeg_paths:
        eeg_data, times, ch_names = load_eeg_data(eeg_path)
        eeg_data = preprocess_eeg_data(eeg_data)

        all_eeg_data.append(eeg_data)

        if all_times is None:
            all_times = times

        if not all_ch_names:
            all_ch_names = ch_names

    if all_times is None:
        all_times = torch.tensor([])

    return torch.stack(all_eeg_data), all_times, all_ch_names
