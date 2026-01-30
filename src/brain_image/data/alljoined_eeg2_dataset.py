from functools import lru_cache
import logging
import re
from typing import Literal, Sequence, cast
import numpy as np
import pandas as pd
from pathlib import Path

import torch

from brain_image.data.data import (
    DSPLIT,
    EEGDataset,
    EEGDatasetConfig,
    EEGDatasetFactory,
    EEGSampleT,
    LatentTypeMapT,
    TensorCache,
    get_eeg_stats,
    get_image_paths,
    rescale_eeg,
    truncate_data,
)


SORT_ORDER = ["subject", "session", "block_id", "sequence_id", "sequence_image_id"]
ALL_SUBS = list(range(1, 21))

@lru_cache
def extract_image_id(image_path: str, re_pattern=re.compile(r".+\/(\d+)\.jpg")):
    m = re_pattern.match(image_path)
    assert m is not None
    return m.groups()[0]


def replace_image_paths(df, new_base: Path) -> pd.Series:
    return df["image_path"].apply(
        lambda p: (new_base / extract_image_id(p)).with_suffix(".jpg")
    )


def load_metadatas(
    eeg_path: Path,
    img_dir: Path,
    subs: list[int],
    sort_order=SORT_ORDER,
    split: Literal["train", "test"] | None = None,
):
    metadatas = pd.concat(
        [
            pd.read_parquet(eeg_path / f"sub-{s:02}" / "experiment_metadata.parquet")
            for s in subs
        ]
    )
    metadatas = metadatas[~metadatas["dropped"]].drop(columns=["dropped", "dataset"])
    metadatas["partition"] = metadatas["partition"].replace(
        {"stim_train": "train", "stim_test": "test"}
    )
    metadatas["image_path"] = replace_image_paths(metadatas, img_dir)
    metadatas = metadatas.sort_values(list(sort_order))

    if split is not None:
        metadatas = metadatas[metadatas["partition"] == split]

    return metadatas


def _load_eeg_from_path(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path, allow_pickle=True)["preprocessed_eeg_data"]).float()


def query_metadata(
    metadatas,
    sub: int | None = None,
    split: Literal["train", "test"] | None = None,
    sort_order: Sequence[str] = SORT_ORDER,
):
    queries = []
    if sub is not None:
        queries.append(f"subject=={sub}")

    if split is not None:
        queries.append(f"partition=='{split}'")

    if queries == []:
        return metadatas

    return metadatas.query(" & ".join(queries)).sort_values(sort_order)


def _get_all_image_paths(img_dir: Path) -> list[Path]:
    return list(img_dir.glob("*.jpg"))


class AlljoinedEEG2DatasetConfig(EEGDatasetConfig):
    data_path: Path = Path("data/alljoined-1.6m")
    img_dir: str = "stimuli/images"
    preprocessed_eeg_dir: str = "preprocessed-eeg"
    dataset: Literal["things-eeg2", "alljoined-eeg2"] = "alljoined-eeg2"
    subs: list[int] | None = None
    num_channels: int = 32
    time_length: int = 250


class AlljoinedEEG2Dataset(EEGDataset):
    def __init__(
        self,
        config: AlljoinedEEG2DatasetConfig,
        split: Literal["train", "val", "test"],
        **kwargs
    ):
        if config.subs is None:
            config.subs = ALL_SUBS

        self.split = split

        data_split = "train" if split == "train" else "test"

        self.img_dir = config.data_path / config.img_dir
        self.metadata = load_metadatas(
            config.data_path / config.preprocessed_eeg_dir,
            self.img_dir,
            config.subs,
            split=data_split,
        )
        
        self.meta_subs: dict[int, pd.DataFrame] = {}
        self.train_eeg_stats = {}


        super().__init__(
            config,
            split,
            **kwargs
        )
        self.config = config

        # Sanity checks
        assert self.eeg.size(1) * self.eeg.size(0) == len(self.metadata), "Number of trials does not match"
        assert self.eeg.size(2) == self.config.num_channels, "Number of channels does not match"
        assert self.eeg.size(3) == self.config.time_length, "Time length does not match"

        for sub in self.config.subs:
            assert len(self.meta_subs[sub]) == self.eeg.size(1), "Number of trials does not match"



    def load_eeg_from_path(self, path: Path):
        return _load_eeg_from_path(path)

    def limit_data_size(self, limit_size: float, limit_shuffle=True):
        if limit_size >= 1.0:
            return

        num_trials = self.eeg.shape[1]
        num_subs = self.eeg.shape[0]

        new_size = int(num_trials * self.limit_size)
        logging.info(
            f"Limiting dataset size to {self.limit_size * 100:.1f}% - {new_size} samples per subject"
        )

        img_idxs = (
            np.random.choice(
                num_trials,
                new_size,
                replace=False,
            )
            if limit_shuffle
            else np.arange(new_size)
        )
        self.eeg = self.eeg[:, img_idxs]  # <sub, trial, time, channel>

        # Map idxs to each subject
        sub_offsets = torch.arange(num_subs) * num_trials
        raw_idxs = img_idxs + sub_offsets[:, None]

        self.metadata = self.metadata.iloc[raw_idxs]  # <sub * trial>

    def get_image_paths(self):
        query_result = self.query_metadata()
        image_paths = query_result["image_path"].to_list()
        image_paths = list(sorted(set(image_paths)))
        return image_paths

    def get_eeg_paths(self, split: DSPLIT | None = None, subs: list[int] | None = None) -> list[Path]:
        if subs is None:
            subs = self.config.subs
        if subs is None:
            subs = ALL_SUBS

        if split is None:
            split = "train" if self.split == "train" else "test"

        eeg_path = self.config.data_path / self.config.preprocessed_eeg_dir
        
        file_name = (
            "preprocessed_eeg_training_flat.npy"
            if split == "train"
            else "preprocessed_eeg_test_flat.npy"
        )

        eeg_paths = [eeg_path / f"sub-{sub:02}" / file_name for sub in subs]
        return eeg_paths


    def query_metadata(self, sub: int | None = None):
        return query_metadata(
            self.metadata, sub
        )

    def __len__(self) -> int:
        return self.eeg.shape[0] * self.eeg.shape[1]

    def prepare(self) -> None:
        for i_sub, sub in enumerate(self.config.subs):
            # Prepare metadata for fast querying
            self.meta_subs[sub] = self.query_metadata(sub).reset_index()
            
            # Prepare EEG stats for rescaling
            train_eeg_paths = self.get_eeg_paths("train", subs=[sub])
            train_eeg = self.load_eeg_from_path(train_eeg_paths[0])
            self.train_eeg_stats[sub] = eeg_stats = get_eeg_stats(train_eeg)
            
            # Preprocess EEG data
            eeg = self.eeg[i_sub] 
            eeg = rescale_eeg(eeg, eeg_stats) 
            eeg = truncate_data(eeg, trunc_max=10)

            self.eeg[i_sub] = eeg


    def __getitem__(self, idx: int) -> EEGSampleT:
        sub, img_idx = divmod(idx, self.eeg.shape[1])

        sub_idx = self.config.subs[sub]
        eeg = self.eeg[sub, img_idx]
        meta = self.meta_subs[sub_idx].iloc[img_idx]
        img_path = Path(meta["image_path"])
        
        sample = {
            "img_path": str(img_path),
            "eeg_data": eeg,
            "idx": idx,
            "sub": sub_idx,
            **self.get_embeddings(img_path),
        }
        return cast(EEGSampleT, sample)



class AlljoinedEEG2DatasetFactory(EEGDatasetFactory):
    def __init__(
        self,
        config: AlljoinedEEG2DatasetConfig,
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
        return AlljoinedEEG2Dataset(
            self.config,
            **kwargs
        )
