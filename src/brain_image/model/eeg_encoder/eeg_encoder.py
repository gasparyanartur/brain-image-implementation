

from abc import ABC
from typing import Literal
from torch import nn, Tensor
import torch

DEFAULT_EEG_DIM = 1024

EEG_ENCODER = Literal["atms", "nice"]

class EEGEncoder(ABC, nn.Module):
    def forward(self, eeg_data: Tensor, sub: Tensor | None = None, *args, **kwargs) -> Tensor:
        raise NotImplementedError()

