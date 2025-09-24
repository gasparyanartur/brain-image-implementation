

from abc import ABC
from torch import nn, Tensor

class EEGEncoder(ABC, nn.Module):
    def forward(self, eeg_data: Tensor, sub: Tensor | None = None, *args, **kwargs) -> Tensor:
        raise NotImplementedError()