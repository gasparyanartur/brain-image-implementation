from functools import lru_cache
import re
from typing import Literal, Sequence
import pandas as pd
from pathlib import Path

from brain_image.data.data import EEGDataset, EEGDatasetConfig, LatentTypeMapT, TensorCache


@lru_cache
def extract_image_id(image_path: str, re_pattern = re.compile(r".+\/(\d+)\.jpg")):
    m = re_pattern.match(image_path)
    assert m is not None
    return m.groups()[0]

def replace_image_paths(df, new_base: Path) -> pd.Series:
    return df["image_path"].apply(lambda p: (new_base / extract_image_id(p)).with_suffix(".jpg"))


def load_metadatas(eeg_path: Path, img_path, subs: list[int]):
    metadatas = pd.concat([
        pd.read_parquet(eeg_path / f"sub-{s:02}" / "experiment_metadata.parquet") for s in subs
    ])
    metadatas = metadatas[~metadatas["dropped"]].drop(columns=["dropped", "dataset"])
    metadatas["partition"] = metadatas["partition"].replace({"stim_train": "train", "stim_test": "test"})
    metadatas["image_path"] = replace_image_paths(metadatas, img_path)
    
    return metadatas

def query_metadata(metadatas, sub: int, split: Literal["train", "test"]):
    return metadatas.query(f"subject=={sub} & partition=='{split}'").sort_values(["sequence_id", "sequence_image_id"])

def _get_all_image_paths(img_dir: Path) -> list[Path]:
    return list(img_dir.glob("*.jpg"))


class AlljoinedEEG2DatasetConfig(EEGDatasetConfig):
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
        # TODO

        super().__init__(config, split, tensor_cache, embeddings_map, standardize_embeddings, limit_size, limit_shuffle, preload_cache)