import logging
from pathlib import Path
import time
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from brain_image.configs import get_device


import torch
from torch import Tensor


from collections.abc import Sequence

from brain_image.utils import current_fig_to_img, z_norm


class IterativeStats:
    def __init__(self, sample_shape: torch.Size | Sequence[int] | int, device: torch.device | str | None = None, unbiased: bool = True):
        if device is None:
            device = get_device()
        if isinstance(device, str):
            device = torch.device(device)

        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        sample_shape = torch.Size(sample_shape)

        self._unbiased = unbiased
        self._sample_shape = sample_shape
        self._count: int = 0
        self._mean: Tensor = torch.zeros(sample_shape, device=device, dtype=torch.float64, requires_grad=False)
        self._mean_2: Tensor = torch.zeros(sample_shape, device=device, dtype=torch.float64, requires_grad=False)

    def update(self, x: Tensor):
        if x.size(0) == 0:
            return

        if x.shape[1:] != self._sample_shape:
            raise ValueError(f"Expected shape {self._sample_shape}, got {x.shape[1:]}")

        x = x.to(dtype=self._mean.dtype, device=self._mean.device)

        batch_n = x.size(0)

        batch_mean = x.mean(dim=0)
        batch_mean_2 = torch.sum((x - batch_mean) ** 2, dim=0)

        delta = batch_mean - self._mean
        count = self._count + batch_n

        mean = self._mean + delta * batch_n / count
        mean_2 = self._mean_2 + batch_mean_2 + delta ** 2 * batch_n * self._count / count

        self._mean = mean
        self._mean_2 = mean_2
        self._count = count

    @property
    def mean(self) -> Tensor:
        return self._mean.float()

    @property
    def std(self) -> Tensor:
        if self._count == 0:
            return torch.zeros_like(self._mean)

        count = self._count - 1 if self._unbiased else self._count
        return torch.sqrt(self._mean_2 / count).float()

    @property
    def stats(self) -> tuple[Tensor, Tensor]:
        return self.mean, self.std
    
    def save_to_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
            
        torch.save({
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
        }, path)


class StatsType(TypedDict):
    mean: Tensor
    std: Tensor


@torch.no_grad()
def plot_projected_latents(
    latents: Sequence[torch.Tensor] | torch.Tensor,
    labels: Sequence[str],
    title: str | None = None,
    pca_dims: int = 50,
):
    pca = PCA(n_components=pca_dims)
    tsne = TSNE(n_components=2)

    if isinstance(latents, torch.Tensor):
        latents = [latents]

    assert len(latents) == len(labels)
    logging.info(f"Projecting {len(latents)} latents to 2D space")

    clean_latents = np.concatenate([z_norm(latent.flatten(1), dim=0).detach().cpu().numpy() for latent in latents], axis=0)

    t1 = time.time()
    clean_latents = pca.fit_transform(clean_latents)
    projected_latents = tsne.fit_transform(clean_latents)
    t2 = time.time()
    logging.info(f"Finished projecting latents in {t2 - t1:.3f} seconds")

    lengths = [latent.size(0) for latent in latents]
    offset = 0
    for label, length in zip(labels, lengths):
        plt.scatter(
            projected_latents[offset : offset + length, 0],
            projected_latents[offset : offset + length, 1],
            label=label,
        )
        offset += length

    plt.legend()
    if title is not None:
        plt.title(title)

    plot_image = current_fig_to_img()
    return plot_image