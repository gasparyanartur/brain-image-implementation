

from abc import ABC
from typing import Literal
from torch import nn, Tensor
import torch

from brain_image.configs import BaseConfig

EEG_ENCODER = Literal["atms", "nice"]

class EEGEncoderConfig(BaseConfig):
    eeg_encoder: EEG_ENCODER
    d_channels: int | None = None
    d_time: int = 250
    d_output: int = 768
    d_eeg: int = 40

class EEGEncoder(ABC, nn.Module):
    def forward(self, eeg_data: Tensor, sub: Tensor | None = None, *args, **kwargs) -> Tensor:
        raise NotImplementedError()

