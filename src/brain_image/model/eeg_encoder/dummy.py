import torch
import torch.nn as nn
from torchvision.models import resnet34
from brain_image.configs import BaseConfig
import einops

from typing import Literal

from brain_image.model.eeg_encoder.eeg_encoder import EEGEncoder, EEGEncoderConfig
from brain_image.model.eeg_encoder.utils import EEGProjection, PatchEmbedding
from brain_image.model.model import ResidualAdd, WrapDebugSequential, is_debug_layer_active


class DummyEEGEncoderConfig(EEGEncoderConfig):
    eeg_encoder: Literal["dummy"] = "dummy"


class DummyEEGEncoder(EEGEncoder):
    def __init__(
        self,
        config: DummyEEGEncoderConfig = DummyEEGEncoderConfig(),
    ):
        # Adapted from https://github.com/eeyhsong/NICE-EEG
        super(DummyEEGEncoder, self).__init__()

        self.config = config
        self.param = nn.Parameter(torch.randn(1))   # Dummy parameter to make the model trainable

        assert config.d_output is not None, "d_output must be specified for NiceEEGEncoder"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape[0], self.config.d_output, device=x.device, dtype=torch.float32)