from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Literal, cast

import numpy as np
import torch
from torch import Tensor, manual_seed
from torch.utils.data import Dataset, random_split
import torchvision
from torchvision.transforms import v2 as tv2
from lightning.pytorch import LightningDataModule

from src.configs import DEFAULT_BATCH_SIZE, BaseConfig, GlobalConfig


class DataConfig(BaseConfig, ABC):
    data_path: Path
    batch_size: int = DEFAULT_BATCH_SIZE
    val_batch_size: int = DEFAULT_BATCH_SIZE
    eval_batch_size: int = DEFAULT_BATCH_SIZE
    shuffle_train: bool = True
    limit_train_size: float = 1.0
    limit_val_size: float = 1.0
    limit_test_size: float = 1.0

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
    latents_dir: str = "img-latents"

    train_imgs_per_concept: int = 10
    test_imgs_per_concept: int = 1
    subs: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Dataloader configuration
    num_workers: int = 8
    eval_batch_size: int = DEFAULT_BATCH_SIZE


class EEGDataModule(DataModule):
    def __init__(self, config: EEGDatasetConfig, model_name: str):
        super().__init__(config)
        self.config: EEGDatasetConfig = config
        self.model_name = model_name

    def get_metadata(self) -> dict:
        return {}

    def get_train_dataset(self) -> EEGDataset:
        return EEGDataset(
            self.config,
            split="train",
            model_name=self.model_name,
        )

    def get_val_dataset(self) -> EEGDataset:
        return EEGDataset(self.config, split="test", model_name=self.model_name)

    def get_test_dataset(self) -> EEGDataset:
        return EEGDataset(self.config, split="test", model_name=self.model_name)

    def train_dataloader(self):
        return self._create_dataloader(
            self.get_train_dataset(),
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
        )

    def val_dataloader(self):
        return self._create_dataloader(
            self.get_val_dataset(), batch_size=self.config.val_batch_size, shuffle=False
        )

    def test_dataloader(self):
        return self._create_dataloader(
            self.get_test_dataset(),
            batch_size=self.config.val_batch_size,
            shuffle=False,
        )

    def _create_dataloader(self, dataset, shuffle=True, batch_size=None):
        if batch_size is None:
            batch_size = self.config.batch_size
        num_workers = getattr(self.config, "num_workers", 8)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


class EEGDataset(Dataset):
    def __init__(
        self,
        config: EEGDatasetConfig,
        split: Literal["train", "test"],
        model_name: str = "synclr",
    ):
        self.config = config
        self.split = split
        self.model_name = model_name
        self.imgs_per_concepts = (
            self.config.train_imgs_per_concept
            if split == "train"
            else self.config.test_imgs_per_concept
        )

        if split == "train":
            img_embed_name = "train_embeddings"
            eeg_name = "preprocessed_eeg_training"

        else:
            img_embed_name = "test_embeddings"
            eeg_name = "preprocessed_eeg_test"

        self.img_paths = get_image_paths(
            self.config.data_path / self.config.imgs_dir,
            split=split,
        )
        self.img_latents = torch.load(
            self.config.data_path
            / self.config.latents_dir
            / model_name
            / f"{img_embed_name}.pt"
        )

        self.eeg_data_paths = [
            self.config.data_path
            / self.config.eeg_dir
            / f"sub-{sub:02}"
            / f"{eeg_name}.npy"
            for sub in self.config.subs
        ]
        self.eeg_data, self.times, self.ch_names = load_all_eeg_data(
            self.eeg_data_paths
        )

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx: int):
        img_idx = idx % (
            len(self.img_paths)
        )  # EEG has stacked over subs, so we need to find the right sample within the sub

        img_path = self.img_paths[img_idx]
        img_latent = self.img_latents[img_idx]
        eeg_data = self.eeg_data[idx]

        return {
            "img_path": str(img_path),
            "img_latent": img_latent,
            "eeg_data": eeg_data,
        }


def prepare_datasets(
    config: EEGDatasetConfig,
    model: str = "synclr",
    seed: int = 42,
    train_val_split: float = 0.8,
    use_test_as_val: bool = True,
) -> tuple[EEGDataset, EEGDataset, EEGDataset]:
    train_dataset = EEGDataset(config, split="train", model_name=model)
    test_dataset = EEGDataset(config, split="test", model_name=model)

    if use_test_as_val:
        val_dataset = EEGDataset(config, split="test", model_name=model)

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
    img = img.float()  # Convert to float tensor
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
    image: torch.Tensor, img_size: tuple[int, int] = (224, 224)
) -> torch.Tensor:
    image = tv2.functional.resize(
        image, list(img_size), interpolation=tv2.InterpolationMode.BICUBIC
    )
    image = image / 255.0
    return image


def preprocess_eeg_data(eeg_data: Tensor) -> Tensor:
    """Preprocess the EEG data by averaging over the number of repetitions.

    Args:
        eeg_data (numpy.ndarray): The EEG data to preprocess. <concepts, repetitions, channels, timesteps>

    Returns:
        numpy.ndarray: The preprocessed EEG data. <concepts, channels, timesteps>
    """
    # Average over the number of repetitions
    preprocessed_data = torch.mean(eeg_data, dim=1)
    return preprocessed_data


def get_image_paths(
    image_dir: Path,
    split: Literal["train", "test"],
    extensions: tuple[str, ...] = (".jpg", ".png"),
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

    return torch.concat(all_eeg_data), all_times, all_ch_names
