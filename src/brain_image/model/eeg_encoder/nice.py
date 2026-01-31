import torch
import torch.nn as nn
from torchvision.models import resnet34
from brain_image.configs import BaseConfig
import einops

from typing import Literal

from brain_image.model.eeg_encoder.eeg_encoder import EEGEncoder, EEGEncoderConfig
from brain_image.model.eeg_encoder.utils import EEGProjection, PatchEmbedding
from brain_image.model.model import ResidualAdd, WrapDebugSequential, is_debug_layer_active


class NiceEEGEncoderConfig(EEGEncoderConfig):
    eeg_encoder: Literal["nice"] = "nice"
    dropout: float = 0.5
    patch_out_size: int = 36
    hidden_dim: int = 1024
    flatten: bool = True


class NiceEEGEncoder(EEGEncoder):
    def __init__(
        self,
        config: NiceEEGEncoderConfig = NiceEEGEncoderConfig(),
    ):
        # Adapted from https://github.com/eeyhsong/NICE-EEG
        super(NiceEEGEncoder, self).__init__(config)

        self.config = config

        assert config.d_channels is not None, "d_channels must be specified for NiceEEGEncoder"

        self.patch_embedding = PatchEmbedding(
            d_embed=config.d_eeg,
            num_channels=config.d_channels,
            dropout=config.dropout,
        )

        self.proj = EEGProjection(
            d_input=config.patch_out_size * config.d_eeg,
            d_output=config.d_output,
            dropout=config.dropout,
        )

        if not config.flatten:
            raise NotImplementedError("Non-flattened case not implemented yet") 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        if x.size(1) != self.config.patch_out_size:
            raise ValueError(f"Expected patch_out_size {self.config.patch_out_size}, got {x.size(1)} for output of size {x.size()}.. Please adjust the patch_out_size in the config.")

        x = x.flatten(start_dim=1)
        x = self.proj(x)

        # TODO: Handle non-flattened case
        
        return x
