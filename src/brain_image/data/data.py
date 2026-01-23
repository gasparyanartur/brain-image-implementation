from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import functools
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Literal, Sequence, TypedDict, Union, cast

import gdown
import numpy as np
import requests
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2 as tv2
from lightning.pytorch import LightningDataModule
import tqdm

from brain_image.configs import BaseConfig, GlobalConfig, get_device_str
import multiprocessing as mp

from brain_image.model.eeg_encoder.eeg_encoder import EEG_ENCODER
from brain_image.model.img_encoder import IMAGE_ENCODER


LatentTypeT = Literal[
    "prior_img_latent", "eeg_latent", "align_img_latent", "low_level_latent"
]
EncoderTypeT = Union[EEG_ENCODER, IMAGE_ENCODER]


class LatentTypeMapT(TypedDict):
    """Maps the role of the latent to the specific encoder used."""

    align_img_latent: IMAGE_ENCODER | None
    prior_img_latent: IMAGE_ENCODER | None
    low_level_latent: IMAGE_ENCODER | None
    eeg_latent: EEG_ENCODER | None


class LatentGroupT(TypedDict):
    """A collection of latents."""

    align_img_latent: torch.Tensor | None
    prior_img_latent: torch.Tensor | None
    low_level_latent: torch.Tensor | None
    eeg_latent: EEG_ENCODER | None


class EEGSampleT(LatentGroupT):
    img_path: str
    idx: int
    sub: int
    eeg_data: torch.Tensor


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
    data_path: Path
    dataset: Literal["things-eeg2", "alljoined-eeg2"]

    preload_cache: bool = True


class TensorCache:
    def __init__(
        self,
        cache_path: Path = Path("tensorcache"),
    ):
        self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=1024 * 1024)
    @staticmethod
    def _load_tensor_from_path(path: Path) -> Tensor:
        tensor = torch.load(path)
        return tensor

    @staticmethod
    def _encode_keys(keys: tuple[str, ...]) -> str:
        return "/".join(keys)

    def _get_tensor_path(self, keys: Sequence[str]) -> Path:
        encoded_path = self._encode_keys(tuple(keys))
        full_path = self.cache_path / encoded_path
        full_path = full_path.with_suffix(".pt")
        return full_path

    def save(self, tensor: torch.Tensor, *keys: str):
        path = self._get_tensor_path(*keys)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)

    def batch_get(
        self, items: Iterable[Iterable[str]], parallel: bool = True
    ) -> torch.Tensor:
        def _get(key_list: list[str]) -> torch.Tensor:
            return self.get(*key_list)

        if parallel:
            with ThreadPoolExecutor() as executor:
                item_list = list(executor.map(_get, items))
        else:
            item_list = [self.get(*item) for item in items]

        return torch.stack(item_list, dim=0)

    def get(self, *keys: str) -> torch.Tensor:
        path = self._get_tensor_path(*keys)
        tensor = self._load_tensor_from_path(path)

        return tensor

    def get_latent(
        self,
        source_path: Path,
        model_name: EncoderTypeT,
        split: Literal["train", "val", "test"],
    ) -> Tensor:
        split = "train" if split == "train" else "test"
        return self.get(str(source_path), model_name, split)


class EEGDatasetFactory(ABC):
    def __init__(
        self,
        config: EEGDatasetConfig,
        tensorcache: TensorCache,
        embeddings_map: dict[LatentTypeT, LatentStats],
    ):
        self.config = config
        self.tensorcache = tensorcache
        self.embeddings_map = embeddings_map

    @abstractmethod
    def create_dataset(
        self, split: Literal["train", "val", "test"], **dataset_kwargs
    ) -> EEGDataset:
        raise NotImplementedError


class DataModule(LightningDataModule, ABC):
    def __init__(self, config: DataConfig, embedding_stats: dict = {}):
        super().__init__()

        self.config = config
        self.num_batches: dict[str, int | None] = {
            "train": None,
            "val": None,
            "test": None,
        }

        self.datasets: dict[str, Dataset | None] = {
            split: None for split in ["train", "val", "test"]
        }
        self.dataloaders: dict[str, torch.utils.data.DataLoader | None] = {}

        self.embedding_stats = embedding_stats or {}

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
        embeddings_map: LatentTypeMapT | None = None,
        embeddings_to_compute_stats: list[IMAGE_ENCODER] | None = None,
    ):
        tensor_cache = tensor_cache or TensorCache()
        embeddings_map = embeddings_map or {
            "align_img_latent": None,
            "prior_img_latent": None,
            "low_level_latent": None,
            "eeg_latent": None,
        }
        embeddings_to_compute_stats = embeddings_to_compute_stats or []

        self.config: EEGDatasetConfig = config
        self.tensor_cache = tensor_cache
        self.embeddings_map = embeddings_map
        self.embeddings_to_compute_stats = embeddings_to_compute_stats

        embeddings_stats = self.get_embeddings_stats()
        logging.info(f"Got embedding stats for: {embeddings_stats.keys()}")

        self.factory = self._create_factory(config, tensor_cache, embeddings_map)

        super().__init__(config, embedding_stats=embeddings_stats)

    def _create_factory(
        self, config, tensor_cache, embeddings_map
    ) -> EEGDatasetFactory:
        match config.factory:
            case "things-eeg2":
                from brain_image.data.things_eeg2_dataset import (
                    ThingsEEG2DatasetFactory,
                )

                return ThingsEEG2DatasetFactory(config, tensor_cache, embeddings_map)  # type: ignore
            case "alljoined":
                raise NotImplementedError
            case _:
                raise ValueError(f"Unrecognized dataset type: {config.dataset}")

    def get_metadata(self) -> dict:
        return {}

    def create_dataset(
        self, split: Literal["train", "val", "test"], **dataset_kwargs
    ) -> EEGDataset:
        return self.factory.create_dataset(split, **dataset_kwargs)

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


class EEGDataset(Dataset):
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
        self.embedding_stats: dict[LatentTypeT, LatentStats] = {}

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
                    img_path, cast(EncoderTypeT, value), self.split
                )
                for key, value in self.embeddings_map.items()
                if value is not None
            },
        )

    def _compute_embedding_stats(self) -> dict[LatentTypeT, LatentStats]:
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
        mapped_stats = cast(dict[LatentTypeT, LatentStats], mapped_stats)
        return mapped_stats

    def _get_image_latent_from_cache(
        self,
        img_path: Path,
        model_name: EncoderTypeT,
        split: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        return self.tensor_cache.get_latent(img_path, model_name, split)

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

    def get_embedding_stats(self) -> dict[LatentTypeT, LatentStats]:
        return self.embedding_stats


def load_image_from_path(path: Path | str, mode: str | None = None) -> Tensor:
    if isinstance(path, str):
        path = Path(path)

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
    paths: Iterable[Path | str],
    parallel: bool = False,
    progressbar: bool = False,
    mode: str | None = None,
) -> Tensor:
    if parallel:
        with ThreadPoolExecutor() as pool:
            imgs = list(
                pool.map(
                    functools.partial(load_image_from_path, mode=mode),
                    paths,
                    timeout=10,
                )
            )
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


class LatentStats(TypedDict):
    mean: Tensor
    std: Tensor
    min: Tensor
    max: Tensor
    norm: Tensor


@torch.no_grad()
def get_embeddings_stats(
    tensorcache: TensorCache,
    img_paths: list[Path],
    embedding_names: list[IMAGE_ENCODER],
    split: Literal["train", "test"],
) -> dict[IMAGE_ENCODER, LatentStats]:
    """Compute a mapping from encoder name to stats.
    E.g. stable_diffusion_v2 -> {mean: 0, std: 1}..."""

    logging.info(
        f"Getting embedding stats for {embedding_names} - {len(img_paths)} images"
    )
    _running_embeddings: dict[IMAGE_ENCODER, Tensor] = {}

    for emb_name in embedding_names:
        arg_list = ((str(img_path), emb_name, split) for img_path in img_paths)
        _running_embeddings[emb_name] = tensorcache.batch_get(arg_list)

    logging.info(f"Keys gathered: {_running_embeddings.keys()}")

    embedding_stats: dict[IMAGE_ENCODER, LatentStats] = {
        k: {
            "mean": torch.mean(v, dim=0),
            "std": torch.std(v, dim=0),
            "min": torch.min(v, dim=0).values,
            "max": torch.max(v, dim=0).values,
            "norm": v.norm(dim=-1).mean(),
        }
        for k, v in _running_embeddings.items()
    }

    logging.info(f"Finished getting embedding stats")
    return embedding_stats


def download_to_file(
    url,
    file_path,
    verbose: bool = True,
    progress_bar: bool = True,
    chunk_size: int = 1024,
    skip_if_exists: bool = True,
    backend: Literal["gdown", "requests"] = "requests",
):
    def _log(s):
        if verbose:
            logging.info(s)

    if skip_if_exists and file_path.exists():
        _log(f"File {file_path} already exists, skipping download")
        return

    _log(f"Downloading file from {url} to {file_path} with backend {backend}")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    match backend:
        case "requests":
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            }
            response = requests.get(url, stream=True, headers=headers)
            total_size = int(response.headers.get("content-length", 0))
            written_size = 0

            _log(f"Using requests backend, saving in {file_path}")
            with open(file_path, "wb") as f:
                with tqdm.tqdm(
                    response.iter_content(chunk_size=chunk_size),
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading",
                    disable=not progress_bar,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        chunk_size = len(chunk)
                        if chunk_size == 0:
                            continue

                        f.write(chunk)
                        written_size += chunk_size
                        pbar.update(chunk_size)

            if total_size > 0 and (written_size != total_size):
                raise ValueError(
                    f"Downloaded size does not match expected size: {written_size} != {total_size}"
                )
        case "gdown":
            _log(f"Using gdown backend, saving in {file_path}")
            gdown.download(output=str(file_path), id=url)
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    if not file_path.exists():
        raise ValueError(f"Failed to download {url} to {file_path}")
    else:
        _log(f"Successfully downloaded {url} to {file_path}")


def merge_data(
    sub: int, img_paths: list[Path], eeg_data: torch.Tensor, idxs: torch.Tensor
) -> list[EEGSampleT]:
    merged_data = []

    for i in range(eeg_data.size(0)):
        idx = idxs[i]
        img_path = img_paths[int(idx)]
        eeg = eeg_data[i]

        joined_object = {"img_path": str(img_path), "eeg": eeg, "sub": sub, "idx": idx}

        merged_data.append(joined_object)

    return merged_data
