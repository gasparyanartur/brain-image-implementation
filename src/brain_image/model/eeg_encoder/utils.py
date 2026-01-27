import torch
from torch import nn
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_time: int = 40,
        d_channel: int = 40,
        d_embed: int = 40,
        num_channels: int = 63,
        kernel_size_time: int = 25,
        kernel_size_pool_time: int = 51,
        stride_pool_time: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.temporal_spatial_conv = nn.Sequential(
            nn.Conv2d(1, d_time, (1, kernel_size_time), stride=(1, 1)),
            nn.AvgPool2d((1, kernel_size_pool_time), (1, stride_pool_time)),
            nn.BatchNorm2d(d_time),
            nn.ELU(),
            nn.Conv2d(d_time, d_channel, (num_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(d_channel),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(d_channel, d_embed, (1, 1), stride=(1, 1)),
        )

        self.rearrange = Rearrange("b e (h) (w) -> b (h w) e")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_spatial_conv(x)
        x = self.projection(x)
        x = self.rearrange(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class EEGProjection(nn.Module):
    def __init__(self, d_input: int, d_output: int, dropout: float):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_input, d_output),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(d_output, d_output),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(d_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
