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

from brain_image.model.img_encoder import IMAGE_ENCODER
from brain_image.utils import flatten_configs


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
    data_path: Path = Path("data") / "things-eeg2"

    imgs_dir: str = "imgs"
    
    prepared_eeg_dir: str = "prepared"      # Needs to be generated with "prepare_data.py"

    subs: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    preload_cache: bool = True

class TensorCache:
    def __init__(
        self,
        cache_path: Path = Path("tensorcache"),
        memory_cache_size: int = 1024*1024,
        use_encrypt: bool = False,
        overwrite: bool = True,
    ):
        self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)

        self._path_cache = {}

        self.memory_cache_size = memory_cache_size
        self.memory_cache = {}
        self.memory_cache_keys = []
        self.use_encrypt = use_encrypt
        self.overwrite = overwrite

    def _get_tensor_path(self, *keys: str) -> Path:
        keys = tuple(keys)
        if keys in self._path_cache:
            return self._path_cache[keys]

        encoded_path = self._encode_keys(*keys, use_encrypt=self.use_encrypt)
        full_path = self.cache_path / encoded_path
        full_path = full_path.with_suffix(".pt")
        self._path_cache[keys] = full_path
        return full_path

    def save(self, tensor: torch.Tensor, *keys: str):
        path = self._get_tensor_path(*keys)
        self._add_to_memory_cache(path, tensor)

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not self.overwrite:
            raise FileExistsError(f"File already exists: {path}")

        torch.save(tensor, path)

    def batch_get(self, items: Iterable[Iterable[str]], parallel: bool = True) -> torch.Tensor:
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
    def __init__(self, config: DataConfig, embedding_stats: dict = {}):
        super().__init__()

        self.config = config
        self.num_batches: dict[str, int | None] = {"train": None, "val": None, "test": None}

        self.datasets: dict[str, Dataset | None] = {split: None for split in ["train", "val", "test"]}
        self.dataloaders: dict[str, torch.utils.data.DataLoader | None] = {}

        self.embedding_stats = embedding_stats or {}

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

    def cleanup(self) -> None:
        dataloader_keys = list(self.dataloaders.keys())
        for key in dataloader_keys:
            if self.dataloaders[key] is None:
                continue
            del self.dataloaders[key]




class EmbeddingsMap(TypedDict):
    align_img_latent: IMAGE_ENCODER | None
    prior_img_latent: IMAGE_ENCODER | None
    low_level_latent: IMAGE_ENCODER | None


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
        tensor_cache: TensorCache = TensorCache(),
        embeddings_map: EmbeddingsMap = {},
        embeddings_to_compute_stats: list[IMAGE_ENCODER] = []
    ):
        self.config: EEGDatasetConfig = config
        self.tensor_cache = tensor_cache
        self.embeddings_map = embeddings_map
        self.embeddings_to_compute_stats = embeddings_to_compute_stats

        embeddings_stats = self._get_embeddings_stats()
        logging.info(f"Got embedding stats for: {embeddings_stats.keys()}")

        super().__init__(config, embedding_stats=embeddings_stats)

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
            limit_shuffle=split=="train",
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

        num_workers = self.config.num_workers or min(32, mp.cpu_count())
        device = get_device_str()
        dataloader_args = {
            "batch_size": self.config.get_batch_size(split),
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": device != "cpu",
            "persistent_workers": (split == "train") and (num_workers > 0),
            "drop_last": False
        }
        dataloader_args.update(kwargs)

        dataset = self._get_dataset(split)
        return torch.utils.data.DataLoader(
            dataset,
            **dataloader_args,
        )

    def _get_embeddings_stats(self):
        img_dir_path = (
            self.config.data_path / self.config.imgs_dir / "training_images"
        )
        img_paths = list(img_dir_path.rglob("*.jpg"))

        embedding_types = [str(v) for k, v in self.embeddings_map.items() if v in self.embeddings_to_compute_stats and v is not None]
        embedding_stats = get_embeddings_stats(
            tensorcache=self.tensor_cache,
            img_paths=img_paths,
            embedding_names=embedding_types,    # type: ignore
            split="train",
        )
        print("EMB STATS", embedding_stats)

        mapped_stats = {k: embedding_stats[v] for k, v in self.embeddings_map.items() if v in embedding_stats}
        return mapped_stats




class EEGDataset(Dataset):
    def __init__(
        self,
        config: EEGDatasetConfig,
        split: Literal["train", "val", "test"],
        tensor_cache: TensorCache,
        embeddings_map: EmbeddingsMap = cast(EmbeddingsMap, {}),
        standardize_embeddings: list[str] = ["prior_img_latent"],
        limit_size: float = 1.0,
        limit_shuffle: bool = True,
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
                    / self.config.prepared_eeg_dir
                    / f"sub-{sub:02}"
                    / f"{split_dir}.pt"
                )
            )
        self.prepared_data = prepared_data
        logging.info(f"Loaded {len(self.prepared_data)} samples")

        if self.limit_size < 1.0:
            new_size = int(len(self.prepared_data) * self.limit_size)
            logging.info(f"Limiting dataset size to {self.limit_size * 100:.1f}% - {new_size} samples")
            
            idxs = np.random.choice(
                len(self.prepared_data),
                new_size,
                replace=False,
            ) if limit_shuffle else np.arange(new_size)
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
            imgs = list(pool.map(functools.partial(load_image_from_path, mode=mode), paths, timeout=10))
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




@torch.no_grad()
def get_embeddings_stats(
    tensorcache: TensorCache,
    img_paths: list[Path],
    embedding_names: list[IMAGE_ENCODER],
    split: Literal["train", "test"],
) -> dict[str, dict[str, torch.Tensor]]:
    logging.info(f"Getting embedding stats for {embedding_names} - {len(img_paths)} images")
    _running_embeddings = {}

    for emb_name in embedding_names:
        arg_list = ((str(img_path), emb_name, split) for img_path in img_paths)
        _running_embeddings[emb_name] = tensorcache.batch_get(arg_list)
    

    logging.info(f"Keys gathered: {_running_embeddings.keys()}")

    _running_latents = _running_embeddings

    logging.info(f"Finished getting embeddings {_running_latents.keys()}")

    embedding_stats: dict[str, dict[str, torch.Tensor]] = {
        k: {
            "mean": torch.mean(v, dim=0),
            "std": torch.std(v, dim=0),
            "min": torch.min(v, dim=0).values,
            "max": torch.max(v, dim=0).values,
            "norm": v.norm(dim=-1).mean(),
        }
        for k, v in _running_latents.items()
    }

    logging.info(f"Finished getting embedding stats")
    return embedding_stats


def download_to_file(
    url,
    file_path,
    verbose: bool = True,
    progress_bar: bool = True,
    chunk_size: int = 1024,
):
    if verbose:
        logging.info(f"Downloading file from {url} to {file_path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get("content-length", 0))
    written_size = 0
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


def merge_data(
    sub: int, img_paths: list[Path], eeg_data: torch.Tensor, idxs: torch.Tensor
) -> list[SampleType]:
    merged_data = []

    for i in range(eeg_data.size(0)):
        idx = idxs[i]
        img_path = img_paths[int(idx)]
        eeg = eeg_data[i]

        joined_object = {"img_path": str(img_path), "eeg": eeg, "sub": sub, "idx": idx}

        merged_data.append(joined_object)

    return merged_data