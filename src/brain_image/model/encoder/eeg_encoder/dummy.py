import torch
import torch.nn as nn

from typing import Literal

from brain_image.model.encoder.eeg_encoder.eeg_encoder import EEGEncoder, EEGEncoderConfig


class DummyEEGEncoderConfig(EEGEncoderConfig):
    eeg_encoder: Literal["dummy"] = "dummy"


class DummyEEGEncoder(EEGEncoder):
    def __init__(
        self,
        config: DummyEEGEncoderConfig = DummyEEGEncoderConfig(),
        **kwargs
    ):
        super(DummyEEGEncoder, self).__init__(config)

        self.config = config
        self.param = nn.Parameter(torch.randn(1))   # Dummy parameter to make the model trainable

        assert config.d_output is not None, "d_output must be specified for NiceEEGEncoder"

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.randn(x.shape[0], self.config.d_output, device=x.device, dtype=torch.float32)