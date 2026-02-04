from __future__ import annotations

from typing import Literal, cast
from pathlib import Path

import torch

from brain_image.data.data import (
    DSPLIT,
    SPLIT,
    EEGSampleT,
    LatentTypeMapT,
)
from brain_image.data.dataset.eeg_dataset import EEGDataset, EEGDatasetConfig
from brain_image.data.tensorcache import TensorCache
from brain_image.model.encoder.img_encoder.union import ImageEncoderName
from brain_image.model.encoder.img_encoder.union import IMAGE_ENCODER_DIM


class DummyEEGDatasetConfig(EEGDatasetConfig):
    dataset: Literal["dummy"] = "dummy"
    data_path: Path = Path("data/dummy")
    subs: list[int] | None = [1]
    num_channels: int = 32
    time_length: int = 250
    dummy_img_path: Path = Path("dummy/img.png")
    dummy_eeg_path: Path = Path("dummy/eeg.npy")
    num_dummy_samples: int = 64

    def create_dataset(self, split: SPLIT, *args, **kwargs) -> DummyEEGDataset:
        return DummyEEGDataset(
            self, split, *args, **kwargs
        )

class DummyEEGDataset(EEGDataset):
    def __init__(
        self,
        config: DummyEEGDatasetConfig,
        split: Literal["train", "val", "test"],
        **kwargs
    ):
        self.split = split

        super().__init__(
            config,
            split,
            **kwargs
        )
        self.config = config

    def load_eeg_from_path(self, path: Path):
        return torch.randn(self.config.num_dummy_samples, self.config.num_channels, self.config.time_length)

    def limit_data_size(self, limit_size: float, limit_shuffle=True):
        ...

    def get_image_paths(self):
        return [self.config.dummy_img_path] * self.config.num_dummy_samples

    def get_eeg_paths(self, split: DSPLIT | None = None, subs: list[int] | None = None) -> list[Path]:
        return [self.config.dummy_eeg_path]

    def _compute_embedding_stats(self):
        stats = {}
        for emb_type, emb_name in self.embeddings_map.items():
            emb_name = cast(ImageEncoderName, emb_name) 
            if emb_name not in self.embeddings_to_compute_stats:
                continue

            dim = IMAGE_ENCODER_DIM[emb_name]
            stats[emb_type] = {
                "mean": torch.randn(dim),
                "std": torch.randn(dim),
            }
        return stats
    

    
    def get_embeddings(self, img_path):
        embedding_stack = {}
        for emb_type, emb_name in self.embeddings_map.items():
            emb_name = cast(ImageEncoderName, emb_name)
            if emb_name is None:
                continue

            dim = IMAGE_ENCODER_DIM[emb_name]
            embedding_stack[emb_type] = torch.randn(dim)
        return embedding_stack

    def __len__(self) -> int:
        assert self.config.subs is not None
        return self.config.num_dummy_samples * len(self.config.subs)

    def prepare(self) -> None:
        ...


    def __getitem__(self, idx: int) -> EEGSampleT:
        sub, img_idx = divmod(idx, self.eeg.shape[1])

        sub_idx = self.config.subs[sub] if self.config.subs is not None else 1
        eeg = self._generate_eeg()
        img_path = self.config.dummy_img_path
        
        sample = {
            "img_path": str(img_path),
            "eeg_data": eeg,
            "idx": idx,
            "sub": sub_idx,
            **self.get_embeddings(img_path),
        }
        return cast(EEGSampleT, sample)

    def _generate_eeg(self):
        return torch.randn(self.config.num_channels, self.config.time_length)


