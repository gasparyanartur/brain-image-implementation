from abc import ABC
from torch import nn, Tensor
import torch
from torch.nn import functional as F
import tqdm
from typing import Any

from brain_image.configs import BaseConfig
from brain_image.utils import gather_records, get_device_from_module


class EEGEncoderConfig(BaseConfig):
    eeg_encoder: str
    d_channels: int | None = None
    d_time: int = 250
    d_output: int = 768
    d_eeg: int = 40


class EEGEncoder(ABC, nn.Module):
    def __init__(self, config: EEGEncoderConfig, *args, **kwargs) -> None:
        super().__init__()

        self.config = config

    def forward(
        self, eeg_data: Tensor, sub: Tensor | None = None, *args, **kwargs
    ) -> Tensor:
        raise NotImplementedError()


def encode_eeg_latent(
    eeg_encoder: EEGEncoder,
    eeg: Tensor,
    subs: Tensor,
) -> dict[str, torch.Tensor]:
    device = get_device_from_module(eeg_encoder)

    eeg = eeg.to(device)
    subs = subs.to(device)
    eeg_latent = eeg_encoder(eeg, subs)
    eeg_latent_normed = F.normalize(eeg_latent)

    return {
        "eeg_latent": eeg_latent, 
        "eeg_latent_normed": eeg_latent_normed
    }


@torch.no_grad()
def batch_encode_eeg_latent(
    eeg_encoder: EEGEncoder,
    eeg: torch.Tensor,
    sub: torch.Tensor,
    batch_size: int = 512,
    progress_bar: bool = True,
) -> dict[str, Any]:

    assert eeg.size(0) == sub.size(
        0
    ), "EEG and subject data must have the same batch size"

    assert eeg is not None, "EEG data is not in batch"
    assert sub is not None, "Subject data is not in batch"

    n = eeg.size(0)

    eeg_batches = [
        encode_eeg_latent(
            eeg_encoder, eeg[i : i + batch_size], sub[i : i + batch_size]
        )
        for i in tqdm.tqdm(
            range(0, n, batch_size), desc="EEG encoding", disable=not progress_bar
        )
    ]

    return gather_records(eeg_batches, tensor_gather="cat")
