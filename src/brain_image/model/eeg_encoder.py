import torch
import torch.nn as nn
from torchvision.models import resnet34
from brain_image.configs import BaseConfig
import einops


from typing import Literal

from brain_image.model.model import ResidualAdd, WrapDebugSequential, is_debug_layer_active


class EEGEncoderConfig(BaseConfig):
    f1: int = 64
    f2: int = 128
    pool1: int = 8
    stride1: int = 5
    pool2: int = 4
    stride2: int = 1
    kernel1: int = 25
    kernel2: int = 17
    dropout: float = 0.5
    embed_dim: int = 128
    patch_out_size: int = 16
    hidden_dim: int = 768
    output_dim: int = 768

    noise_augment: float = 0.00
    temporal_zero_prob: float = 0.0
    spatial_zero_prob: float = 0.0
    #noise_augment: float = 0.01
    #temporal_zero_prob: float = 0.2
    #spatial_zero_prob: float = 0.1
    norm_type: Literal["batch", "group"] = "group"
    norm_groups: int = 16


class EEGEncoder(nn.Module):
    def __init__(
        self,
        config: EEGEncoderConfig = EEGEncoderConfig(),
        norm_func: type[nn.Module] = nn.BatchNorm2d,
        act_func: type[nn.Module] = nn.ELU,
    ):
        # Adapted from https://github.com/eeyhsong/NICE-EEG
        super(EEGEncoder, self).__init__()

        self.config = config

        if config.norm_type == "batch":
            norm = nn.BatchNorm2d
        elif config.norm_type == "group":
            norm = lambda c: nn.GroupNorm(num_groups=config.norm_groups, num_channels=c)
        else:
            raise ValueError(f"Unknown norm_type: {config.norm_type}")

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                1,
                config.f1,
                kernel_size=(1, config.kernel1),
                bias=False,
                stride=(1, config.stride1),
            ),
            norm(config.f1),
            act_func(),
            nn.Conv2d(
                config.f1,
                config.f2,
                kernel_size=(config.kernel2, 1),
                bias=False,
            ),
            norm(config.f2),
            act_func(),
            nn.Dropout(config.dropout, inplace=True),
            nn.Conv2d(config.f2, config.embed_dim, kernel_size=1),
        )

        if is_debug_layer_active("eeg_layers"):
            self.patch_embedding = WrapDebugSequential(
                self.patch_embedding, "patch_embedding"
            )

        self.proj = nn.Sequential(
            #nn.LayerNorm(config.embed_dim * config.patch_out_size),
            nn.Linear(config.patch_out_size * config.embed_dim, config.hidden_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                )
            ),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, T = x.shape

        if self.training:
            if self.config.noise_augment > 0:
                x += torch.randn_like(x) * self.config.noise_augment

            if self.config.temporal_zero_prob > 0:
                num_temporal = x.shape[-1]
                zero_mask = torch.rand(num_temporal) < self.config.temporal_zero_prob
                zero_mask = (
                    zero_mask[None, None, :].repeat(B, S, 1).to(x.device, dtype=x.dtype)
                )
                x = x * (1 - zero_mask)

            if self.config.spatial_zero_prob > 0:
                num_spatial = x.shape[-2]
                zero_mask = torch.rand(num_spatial) < self.config.spatial_zero_prob
                zero_mask = (
                    zero_mask[None, :, None].repeat(B, 1, T).to(x.device, dtype=x.dtype)
                )
                x = x * (1 - zero_mask)

        x = einops.rearrange(x, "b s t -> b 1 s t")
        x = self.patch_embedding(x)
        x = einops.rearrange(x, "b e s t -> b (s t e)")
        x = self.proj(x)
        return x


class EEGEncoder2(nn.Module):
    def __init__(self, config: EEGEncoderConfig = EEGEncoderConfig()):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 25), stride=(1, 5)),   # 13, 15
            nn.GroupNorm(8, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(1, 2)),    # 11, 6
            nn.GroupNorm(16, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),   # 9, 4
            nn.GroupNorm(32, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=(9, 4)),
            nn.Flatten(),
        )

        self.seq = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            ResidualAdd(
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    )
            ),
            nn.Linear(512, config.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b s t -> b 1 s t")
        x = self.convs(x)
        x = self.seq(x)
        return x


class EEGEncoder3(nn.Module):
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