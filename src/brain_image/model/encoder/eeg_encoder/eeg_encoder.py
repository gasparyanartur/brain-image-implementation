from abc import ABC
from torch import nn, Tensor

from brain_image.configs import BaseConfig


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
