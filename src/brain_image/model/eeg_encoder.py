import torch
import torch.nn as nn
from torchvision.models import resnet34
from brain_image.configs import BaseConfig
import einops

from typing import Literal

from brain_image.model.model import ResidualAdd, WrapDebugSequential, is_debug_layer_active


class EEGEncoderConfig(BaseConfig):
    f1: int = 40
    f2: int = 40
    pool1: int = 8
    stride1: int = 5
    pool2: int = 4
    stride2: int = 1
    kernel1: int = 25
    kernel2: int = 17
    dropout: float = 0.5
    embed_dim: int = 40
    patch_out_size: int = 36
    hidden_dim: int = 1024
    output_dim: int = 768


class PatchEmbedding(nn.Module):
    def __init__(self, feat_t=40, kern_t=25, kern_pool_t=51, stride_pool_t=5, feat_s=40, kern_s=63, embed_dim=40, dropout=0.5):
        super().__init__()
        self.temporal_spatial_conv = nn.Sequential(
            nn.Conv2d(1, feat_t, (1, kern_t), stride=(1, 1)),
            nn.AvgPool2d((1, kern_pool_t), (1, stride_pool_t)),
            nn.BatchNorm2d(feat_t),
            nn.ELU(),
            nn.Conv2d(feat_t, feat_s, (kern_s, 1), stride=(1, 1)),
            nn.BatchNorm2d(feat_s),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(feat_s, embed_dim, (1, 1), stride=(1, 1)),  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)     
        x = self.temporal_spatial_conv(x)
        x = self.projection(x)
        return x


class NiceEEGEncoder(nn.Module):
    def __init__(
        self,
        config: EEGEncoderConfig = EEGEncoderConfig(),
    ):
        # Adapted from https://github.com/eeyhsong/NICE-EEG
        super(NiceEEGEncoder, self).__init__()

        self.config = config

        self.patch_embedding = PatchEmbedding(
            embed_dim=config.embed_dim,
            dropout=config.dropout
        )

        self.proj = nn.Sequential(
            nn.Linear(config.patch_out_size * config.embed_dim, config.output_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(config.output_dim, config.output_dim),
                    nn.Dropout(config.dropout)
                )
            ),
            nn.LayerNorm(config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = x.flatten(start_dim=1)
        x = self.proj(x)
        return x


class EEGEncoderResnet(nn.Module):
    def __init__(self, config: EEGEncoderConfig = EEGEncoderConfig()):
        super().__init__()

        model = resnet34(weights="DEFAULT")
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(1, 25), stride=(1, 3), padding=(0, 0), bias=False)
        model.fc = torch.nn.Linear(512, 768)
        model.max_pool = torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), dilation=1, ceil_mode=False)

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b s t -> b 1 s t")
        x = self.model(x)
        return x