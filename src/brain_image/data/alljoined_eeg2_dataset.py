from functools import lru_cache
import logging
import re
from typing import Literal, Sequence
import numpy as np
import pandas as pd
from pathlib import Path

import torch

from brain_image.data.data import EEGDataset, EEGDatasetConfig, LatentTypeMapT, TensorCache


@lru_cache
def extract_image_id(image_path: str, re_pattern = re.compile(r".+\/(\d+)\.jpg")):
    m = re_pattern.match(image_path)
    assert m is not None
    return m.groups()[0]

def replace_image_paths(df, new_base: Path) -> pd.Series:
    return df["image_path"].apply(lambda p: (new_base / extract_image_id(p)).with_suffix(".jpg"))


def load_metadatas(eeg_path: Path, img_dir: Path, subs: list[int], sort_order=(
        "subject", "session", "block_id", "sequence_id", "sequence_image_id"
)):
    metadatas = pd.concat([
        pd.read_parquet(eeg_path / f"sub-{s:02}" / "experiment_metadata.parquet") for s in subs
    ])
    metadatas = metadatas[~metadatas["dropped"]].drop(columns=["dropped", "dataset"])
    metadatas["partition"] = metadatas["partition"].replace({"stim_train": "train", "stim_test": "test"})
    metadatas["image_path"] = replace_image_paths(metadatas, img_dir)
    metadatas = metadatas.sort_values(list(sort_order))

    return metadatas

def _load_eeg_from_path(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path, allow_pickle=True)["preprocessed_eeg_data"])

def load_egg_values(eeg_path: Path, subs: list[int], split: Literal["train", "test"]) -> torch.Tensor:
    file_name = "preprocessed_eeg_training_flat.npy" if split == "train" else "preprocessed_eeg_test_flat.npy"
    
    eeg_paths = [eeg_path / f"sub-{sub:02}" / file_name for sub in subs]
    eeg = torch.stack([_load_eeg_from_path(eeg_path) for eeg_path in eeg_paths])  # <sub, image, channel, space, time>

    return eeg

def query_metadata(metadatas, sub: int, split: Literal["train", "test"]):
    return metadatas.query(f"subject=={sub} & partition=='{split}'").sort_values(["sequence_id", "sequence_image_id"])

def _get_all_image_paths(img_dir: Path) -> list[Path]:
    return list(img_dir.glob("*.jpg"))


class AlljoinedEEG2DatasetConfig(EEGDatasetConfig):
    data_path: Path = Path("data/alljoined-1.6m")
    img_dir: str = "stimuli/images"
    preprocessed_eeg_dir: str = "preprocessed-eeg"
    dataset: Literal['things-eeg2', 'alljoined-eeg2'] = 'alljoined-eeg2'
    subs: list[int] = list(range(1, 21))

class AlljoinedEEG2Dataset(EEGDataset):
    def __init__(
            self, 
            config: AlljoinedEEG2DatasetConfig, 
            split: Literal["train", "val", "test"],
            tensor_cache: TensorCache | None = None,
            embeddings_map: LatentTypeMapT | None = None,
            standardize_embeddings: Sequence[str] = ("prior_img_latent",),
            limit_size: float | None = None,
            limit_shuffle: bool = True,
            preload_cache: bool | None = None,
        ):
        self.img_dir = config.data_path / config.img_dir
        self.img_paths = _get_all_image_paths(self.img_dir)
        self.metadata = load_metadatas(config.data_path / config.preprocessed_eeg_dir, self.img_dir, config.subs)
        self.eeg = load_egg_values(config.data_path / config.preprocessed_eeg_dir, subs=config.subs, split="train" if split == "train" else "test")

        super().__init__(config, split, tensor_cache, embeddings_map, standardize_embeddings, limit_size, limit_shuffle, preload_cache)

    def limit_data_size(self, limit_size: float, limit_shuffle = True):
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
        self.metadata = self.metadata.iloc[idxs]

    def query_metadata(self, sub: int):
        return query_metadata(self.metadata, sub, "train" if self.split == "train" else "test")
    
    def __len__(self):
        return self.eeg.shape[0]