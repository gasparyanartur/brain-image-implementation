from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Mapping, Type, TypeVar, TypedDict, Union, cast

import mne
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torchvision.transforms import v2 as tv2

from brain_image.data.tensorcache import TensorCache
from brain_image.stats import StatsType

T = TypeVar('T')

SPLIT = Literal["train", "val", "test"]
DSPLIT = Literal["train", "test"]

DEFAULT_IMAGE_PIPE = tv2.Compose([
    tv2.ToImage(),
    tv2.ToDtype(torch.float32, scale=True),
])
def resize_crop_image(image: Tensor, size: int = 224) -> Tensor:
    h, w = image.shape[-2:]

    if w > h:
        h = size
        w = int(size * w / h)
    else:
        w = size
        h = int(size * h / w)

    img = tv2.functional.resize(image, size=[h, w], antialias=True, interpolation=tv2.InterpolationMode.BICUBIC)
    img = tv2.functional.center_crop(img, [size, size])
    return img

def preprocess_image(image: Tensor, pipe: nn.Module = DEFAULT_IMAGE_PIPE, size: int = 224) -> Tensor:
    image = pipe(image)
    image = resize_crop_image(image, size=size)
    return image

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


@torch.no_grad()
def load_stats(
    stat_dir: Path, dataset_name, split, stat_name: str
) -> StatsType:
    stat_path = stat_dir / "datasets" / dataset_name / split / f"{stat_name}.pt"
    stats = torch.load(stat_path)
    return stats

@torch.no_grad()
def old_get_embeddings_stats(
    tensorcache: TensorCache,
    img_paths: list[Path],
    embedding_names: list[str],
    split: Literal["train", "test"],
) -> dict[str, StatsType]:
    """Compute a mapping from encoder name to stats.
    E.g. stable_diffusion_v2 -> {mean: 0, std: 1}..."""

    logging.info(
        f"Getting embedding stats for {embedding_names} - {len(img_paths)} images"
    )
    _running_embeddings: dict[str, Tensor] = {}

    for emb_name in embedding_names:
        arg_list = ((str(img_path), emb_name, split) for img_path in img_paths)
        _running_embeddings[emb_name] = tensorcache.batch_get(arg_list)

    logging.info(f"Keys gathered: {_running_embeddings.keys()}")

    embedding_stats: dict[str, StatsType] = {
        k: {
            "mean": torch.mean(v, dim=0),
            "std": torch.std(v, dim=0),
        }
        for k, v in _running_embeddings.items()
    }

    logging.info(f"Finished getting embedding stats")
    return embedding_stats


@torch.no_grad()
def get_eeg_stats(eeg: Tensor) -> dict[str, Tensor]:

    num_channels = eeg.size(-2)
    num_timesteps = eeg.size(-1)

    eeg = eeg.reshape(-1, num_channels, num_timesteps)
    mean = eeg.mean(dim=0)
    std = eeg.std(dim=0)

    return {
        "mean": mean,
        "std": std,
    }


@torch.no_grad()
def rescale_eeg(eeg: Tensor, stats: dict[str, Tensor], reverse: bool = False) -> Tensor:
    if reverse:
        eeg = eeg * stats["std"] + stats["mean"]
    else:
        eeg = (eeg - stats["mean"]) / stats["std"]
    return eeg


@torch.no_grad()
def truncate_data(
    data: torch.Tensor,
    trunc_percentile: float | None = None,
    trunc_max: float | None = None,
):
    if trunc_percentile is None and trunc_max is None:
        return data

    if trunc_percentile is not None and trunc_max is not None:
        raise ValueError(f"Only one of trunc_percentile or trunc_max can be specified.")

    if trunc_max is None:
        assert trunc_percentile is not None
        sorted_values = data.flatten().abs().sort(descending=False).values
        trunc_idx = int(trunc_percentile * len(sorted_values))
        trunc_max = sorted_values[trunc_idx].item()

    data[data > trunc_max] = trunc_max
    data[data < -trunc_max] = -trunc_max
    return data


def merge_data(
    sub: int, img_paths: list[Path], eeg_data: torch.Tensor, idxs: torch.Tensor
) -> list[dict[str, Any]]:
    merged_data = []

    for i in range(eeg_data.size(0)):
        idx = idxs[i]
        img_path = img_paths[int(idx)]
        eeg = eeg_data[i]

        joined_object = {"img_path": str(img_path), "eeg": eeg, "sub": sub, "idx": idx}

        merged_data.append(joined_object)

    return merged_data


def get_from_batch(key: str, batch: Mapping[str, Any], type_: Type[T]) -> T:
    assert (val := batch.get(key)) is not None, f"{key} is not in batch"
    assert isinstance(val, type_), f"{key} is not of type {type_}"
    return cast(T, val)


def get_channel_coords_from_names(ch_names: list[str], montage_type: str = "standard_1020") -> Tensor:
    montage = mne.channels.make_standard_montage(montage_type)
    montage_positions = montage.get_positions()["ch_pos"]
    
    coords = np.stack([montage_positions[ch] for ch in ch_names], axis=0)  # (num_channels, 3)
    coords = torch.from_numpy(coords).float()  # (num_channels, 3)
    return coords