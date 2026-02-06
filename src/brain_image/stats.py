from pathlib import Path
from typing import TypedDict
from brain_image.configs import get_device


import torch
from torch import Tensor


from collections.abc import Sequence


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