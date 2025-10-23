

from abc import ABC
from torch import nn, Tensor
import torch

DEFAULT_EEG_DIM = 1024

class EEGEncoder(ABC, nn.Module):
    def forward(self, eeg_data: Tensor, sub: Tensor | None = None, *args, **kwargs) -> Tensor:
        raise NotImplementedError()

