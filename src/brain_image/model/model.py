from __future__ import annotations
import os
from pathlib import Path

import torch
import torch.nn as nn

from brain_image.configs import BaseConfig

from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

@torch.compile()
def normalize_projection(
    x: torch.Tensor, rescale_norm_by_mean: bool = False
) -> torch.Tensor:
    if rescale_norm_by_mean:
        x = x - x.mean(dim=-1, keepdim=True)
    return nn.functional.normalize(x, dim=-1, p=2)


class DebugLayer(nn.Module):
    def __init__(self, note: str = ""):
        super(DebugLayer, self).__init__()
        self.note = note

    def forward(self, x):
        print(f"(debug): {x.shape} - {self.note}")
        return x


class WrapDebugSequential(nn.Module):
    def __init__(self, seq: nn.Sequential, note: str = ""):
        super(WrapDebugSequential, self).__init__()
        self.seq = seq

        self.debug_layers = nn.ModuleList(
            [DebugLayer(f"layer_{i}") for i in range(len(seq))]
        )

    def forward(self, x):
        for i, layer in enumerate(self.seq):
            x = layer(x)
            print(f"(debug): {i} - {x.shape} - {self.debug_layers[i].note}")
        return x

class LinearLayerNorm(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, act_func: nn.Module | None = None, do_layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.act_func = act_func
        self.do_layer_norm = do_layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.do_layer_norm:
            x = self.norm(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x

    

class ResidualAdd(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class ResidualAdapter(nn.Module):
    def __init__(self, latent_dim: int = 768, hidden_factor: int = 2, dropout: float = 0.5):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim * hidden_factor),
            nn.GELU(),
            nn.Linear(latent_dim * hidden_factor, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


def is_debug_layer_active(layer_name: str) -> bool:
    query = "DEBUG_" + layer_name.upper().strip()
    result = os.getenv(query)
    if result is None:
        result = os.getenv(query.lower())
        if result is None:
            return False

    result = result.lower().strip()
    if result in {"1", "true", "yes"}:
        return True
    elif result in {"0", "false", "no"}:
        return False
    else:
        raise ValueError(f"Invalid value for {layer_name}: {result}")


class LatentProjector(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        proj_dim: int = 768,
        hidden_dim: int = 768,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.l_proj = nn.Linear(embed_dim, hidden_dim)
        self.l_inner = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.l_out = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x = self.l_proj(x)
        x = self.l_inner(x) + x_res
        x = self.norm1(x)
        x = self.l_out(x)

        return x



class TrainingModuleConfig(BaseConfig):
    max_epochs: int = 100
    seed: int = 42
    log_on_step: bool = False


class TrainingModule(pl.LightningModule):
    def __init__(self, config: TrainingModuleConfig):
        super().__init__()
        self.config = config

    @property
    def log_dir(self) -> Path | None:
        for logger in self.loggers:
            if isinstance(logger, (CSVLogger, TensorBoardLogger)):
                if logger.log_dir is not None:
                    return Path(logger.log_dir)

        return None