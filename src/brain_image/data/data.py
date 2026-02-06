from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Mapping, Type, TypeVar, TypedDict, Union, cast

import numpy as np
import torch
from torch import Tensor


from brain_image.data.tensorcache import TensorCache
from brain_image.model.encoder.eeg_encoder.union import EEGEncoderName
from brain_image.model.encoder.img_encoder.union import ImageEncoderName

T = TypeVar('T')

SPLIT = Literal["train", "val", "test"]
DSPLIT = Literal["train", "test"]

class LatentTypeMapT(TypedDict):
    """Maps the role of the latent to the specific encoder used."""

    align_img_latent: ImageEncoderName | None
    prior_img_latent: ImageEncoderName | None
    low_level_latent: ImageEncoderName | None
    eeg_latent: EEGEncoderName | None


class LatentGroupT(TypedDict):
    """A collection of latents."""

    align_img_latent: torch.Tensor | None
    prior_img_latent: torch.Tensor | None
    low_level_latent: torch.Tensor | None
    eeg_latent: EEGEncoderName | None


class EEGSampleT(LatentGroupT):
    img_path: str
    idx: int
    sub: int
    eeg_data: torch.Tensor


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


class LatentStats(TypedDict):
    mean: Tensor
    std: Tensor


@torch.no_grad()
def get_embeddings_stats(
    tensorcache: TensorCache,
    img_paths: list[Path],
    embedding_names: list[ImageEncoderName],
    split: Literal["train", "test"],
) -> dict[ImageEncoderName, LatentStats]:
    """Compute a mapping from encoder name to stats.
    E.g. stable_diffusion_v2 -> {mean: 0, std: 1}..."""

    logging.info(
        f"Getting embedding stats for {embedding_names} - {len(img_paths)} images"
    )
    _running_embeddings: dict[ImageEncoderName, Tensor] = {}

    for emb_name in embedding_names:
        arg_list = ((str(img_path), emb_name, split) for img_path in img_paths)
        _running_embeddings[emb_name] = tensorcache.batch_get(arg_list)

    logging.info(f"Keys gathered: {_running_embeddings.keys()}")

    embedding_stats: dict[ImageEncoderName, LatentStats] = {
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
) -> list[EEGSampleT]:
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