import logging
from typing import Literal, Sequence

import numpy as np
import torch
from brain_image.data.data import (
    EEGDataset,
    EEGDatasetConfig,
    EEGSampleT,
    LatentTypeMapT,
    TensorCache,
)


class ThingsEEG2DatasetConfig(EEGDatasetConfig): ...


class ThingsEEG2Dataset(EEGDataset):
    def __init__(
        self,
        config: ThingsEEG2DatasetConfig,
        split: Literal["train", "val", "test"],
        tensor_cache: TensorCache | None = None,
        embeddings_map: LatentTypeMapT | None = None,
        standardize_embeddings: Sequence[str] = ("prior_img_latent",),
        limit_size: float = 1.0,
        limit_shuffle: bool = True,
        preload_cache: bool = True,
    ):
        super().__init__(
            config,
            split,
            tensor_cache,
            embeddings_map,
            standardize_embeddings,
            limit_size,
            limit_shuffle,
            preload_cache,
        )

    def prepare(self) -> None:
        prepared_data: list[dict] = []
        split_dir = "train" if self.split == "train" else "test"
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

    def limit_data_size(self, limit_size: float, limit_shuffle: bool = True) -> None:
        if limit_size >= 1.0:
            return

        new_size = int(len(self.prepared_data) * self.limit_size)
        logging.info(
            f"Limiting dataset size to {self.limit_size * 100:.1f}% - {new_size} samples"
        )

        idxs = (
            np.random.choice(
                len(self.prepared_data),
                new_size,
                replace=False,
            )
            if limit_shuffle
            else np.arange(new_size)
        )
        self.prepared_data = [self.prepared_data[i] for i in idxs]

    def __len__(self) -> int:
        return len(self.prepared_data)

    def __getitem__(self, idx: int) -> EEGSampleT:
        item = self.prepared_data[idx]

        sample = {
            "img_path": str(item["img_path"]),
            "eeg_data": item["eeg"],
            "idx": item["idx"],
            "sub": item["sub"],
            **self._get_embeddings(item["img_path"]),
        }

        return sample