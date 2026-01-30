import logging
from pathlib import Path
from typing import Literal, Sequence, cast

import numpy as np
import torch
from torch import Tensor
from brain_image.data.data import (
    DSPLIT,
    EEGDataset,
    EEGDatasetConfig,
    EEGDatasetFactory,
    EEGSampleT,
    LatentStats,
    LatentTypeMapT,
    LatentTypeT,
    TensorCache,
    get_image_paths,
)


ALL_SUBS = list(range(1, 11))


def _load_eeg_from_path(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path, allow_pickle=True)["preprocessed_eeg_data"]).float()

def load_eeg_values(eeg_path: Path, subs: list[int], split: Literal["train", "test"]) -> torch.Tensor:
    file_name = "training.npy" if split == "train" else "test.npy"
    eeg_paths = [
            eeg_path
            / f"sub-{sub:02}"
            / file_name
        for sub in subs
    ] 
    eeg = torch.stack([_load_eeg_from_path(eeg_path) for eeg_path in eeg_paths])  # <sub, image, channel, space, time>
    return eeg


class ThingsEEG2DatasetConfig(EEGDatasetConfig):
    data_path: Path = Path("data/things-eeg2")
    img_dir: str = "imgs"
    preprocessed_eeg_dir: str = "preprocessed-eeg"
    dataset: Literal['things-eeg2', 'alljoined'] = "things-eeg2"
    subs: list[int] | None = None
    num_channels: int = 63
    time_length: int = 250


class ThingsEEG2Dataset(EEGDataset):
    def __init__(
        self,
        config: ThingsEEG2DatasetConfig,
        split: Literal["train", "val", "test"],
        **kwargs
    ):
        if config.subs is None:
            config.subs = ALL_SUBS

        self.img_dir = config.data_path / config.img_dir
        self.img_paths = get_image_paths(
            self.img_dir,
            split="train" if split == "train" else "test",
            extensions=(".jpg",),
        )
         
        super().__init__(
            config,
            split,
            **kwargs
        )
        self.config = config

    def prepare(self) -> None:
        ...

    def get_image_paths(self):
        return self.img_paths
    
    def load_eeg_from_path(self, path: Path) -> Tensor:
        eeg = np.load(path, allow_pickle=True)["preprocessed_eeg_data"]
        eeg = torch.from_numpy(eeg)
        eeg = eeg.mean(dim=1)
        return eeg

    def get_eeg_paths(self, split: DSPLIT | None = None, subs: list[int] | None = None) -> list[Path]:
        if subs is None:
            subs = self.config.subs
        if subs is None:
            subs = ALL_SUBS

        if split is None:
            split = "train" if self.split == "train" else "test"

        eeg_path = self.config.data_path / self.config.preprocessed_eeg_dir
        
        file_name = "training.npy" if split == "train" else "test.npy"
        eeg_paths = [
                eeg_path
                / f"sub-{sub:02}"
                / file_name
            for sub in subs
        ] 
        return eeg_paths


    def limit_data_size(self, limit_size: float, limit_shuffle: bool = True) -> None:
        if limit_size >= 1.0:
            return

        new_size = int(len(self) * self.limit_size)
        logging.info(
            f"Limiting dataset size to {self.limit_size * 100:.1f}% - {new_size} samples"
        )

        idxs = (
            np.random.choice(
                len(self),
                new_size,
                replace=False,
            )
            if limit_shuffle
            else np.arange(new_size)
        )
        self.eeg = self.eeg[:, idxs]
        self.img_paths = [self.img_paths[i] for i in idxs]

    def __len__(self) -> int:
        return self.eeg.shape[0] * self.eeg.shape[1]

    def __getitem__(self, idx: int) -> EEGSampleT:
        sub, img_idx = divmod(idx, self.eeg.shape[1])

        sub_idx = self.config.subs[sub]     # type: ignore
        img_path = self.img_paths[img_idx]
        sample = {
            "img_path": str(img_path),
            "eeg_data": self.eeg[sub, img_idx],
            "idx": idx,
            "sub": sub_idx,
            **self.get_embeddings(img_path),
        }

        return cast(EEGSampleT, sample)


class ThingsEEG2DatasetFactory(EEGDatasetFactory):
    def __init__(
        self,
        config: ThingsEEG2DatasetConfig,
        tensorcache: TensorCache,
        embeddings_map: LatentTypeMapT,
    ):
        self.config = config
        self.tensorcache = tensorcache
        self.embeddings_map = embeddings_map

    def create_dataset(
        self, split: Literal["train", "val", "test"], **dataset_kwargs
    ) -> EEGDataset:
        kwargs = {
            "split": split,
            "tensor_cache": self.tensorcache,
            "embeddings_map": self.embeddings_map,
            "limit_size": self.config.get_limit_size(split),
            "limit_shuffle": split == "train",
            "preload_cache": self.config.preload_cache,
        }
        kwargs.update(dataset_kwargs)
        return ThingsEEG2Dataset(
            self.config,
            **kwargs
        )
