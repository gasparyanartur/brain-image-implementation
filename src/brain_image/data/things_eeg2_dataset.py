import logging
from typing import Literal, Sequence, cast

import numpy as np
import torch
from brain_image.data.data import (
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


class ThingsEEG2DatasetConfig(EEGDatasetConfig):
    img_dir: str = "imgs"
    preprocessed_eeg_dir: str = "preprocessed-eeg"
    dataset: Literal['things-eeg2', 'alljoined'] = "things-eeg2"


class ThingsEEG2Dataset(EEGDataset):
    def __init__(
        self,
        config: ThingsEEG2DatasetConfig,
        split: Literal["train", "val", "test"],
        sub: int,
        tensor_cache: TensorCache | None = None,
        embeddings_map: LatentTypeMapT | None = None,
        standardize_embeddings: Sequence[str] = ("prior_img_latent",),
        limit_size: float | None = None,
        limit_shuffle: bool = True,
        preload_cache: bool | None = None,
    ):
        self.img_dir = config.data_path / config.img_dir
        self.img_paths = get_image_paths(
            self.img_dir,
            split="train" if split == "train" else "test",
            extensions=(".jpg",),
        )

        self.eeg_path = (
            config.data_path
            / config.preprocessed_eeg_dir
            / f"sub-{sub:02}"
            / f"{'training' if split == "train" else 'test'}.npy"
        )

        self.eeg: torch.Tensor = torch.from_numpy(np.load(self.eeg_path, allow_pickle=True)["preprocessed_eeg_data"])

        super().__init__(
            config,
            split,
            sub,
            tensor_cache,
            embeddings_map,
            standardize_embeddings,
            limit_size,
            limit_shuffle,
            preload_cache,
        )

    def prepare(self) -> None:
        self.eeg: torch.Tensor = self.eeg.mean(dim=1)

    def get_image_paths(self):
        return self.img_paths

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
        self.eeg = self.eeg[idxs]
        self.img_paths = [self.img_paths[i] for i in idxs]

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, idx: int) -> EEGSampleT:
        img_path = self.img_paths[idx]
        sample = {
            "img_path": str(img_path),
            "eeg_data": self.eeg[idx],
            "idx": idx,
            "sub": self.sub,
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
        return ThingsEEG2Dataset(
            self.config,
            split=split,
            tensor_cache=self.tensorcache,
            embeddings_map=self.embeddings_map,
            limit_size=self.config.get_limit_size(split),
            limit_shuffle=split == "train",
            preload_cache=self.config.preload_cache,
        )
