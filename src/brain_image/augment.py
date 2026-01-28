from __future__ import annotations

from torch import Tensor
import torch
from torch import nn
from torchvision.transforms import v2 as tv2
from typing import Any


class AugmentAmplitudeScale(nn.Module):
    def __init__(self, min_scale: float = 0.5, max_scale: float = 2, prob: float = 0.2):
        super().__init__()

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        # x: <B, C, T>
        assert x.ndim == 3
        B, C, T = x.shape
        dvc = x.device

        nonmask = torch.rand(B, device=dvc) > self.prob
        scale = (
            torch.rand((B, 1, 1), device=dvc) * (self.max_scale - self.min_scale)
            + self.min_scale
        )
        scale[nonmask] = 1

        x = x * scale
        return x


class AugmentTimeShift(nn.Module):
    def __init__(self, max_shift_scale: float = 0.2, prob: float = 0.2):
        super().__init__()

        self.max_shift_scale = max_shift_scale
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        # x: <B, C, T>
        assert x.ndim == 3
        B, C, T = x.shape
        dvc = x.device

        # 1. Randomly select shifts for each sample
        # 2. Set shifts to 0 with probability 1 - prob
        # 3. Roll the T dimension by the shift for each sample

        shift_scale = int(T * self.max_shift_scale)

        nonmask = torch.rand((B,), device=dvc) > self.prob
        shifts = torch.randint(
            low=-shift_scale,
            high=shift_scale,
            size=(B,),
            device=dvc,
        )

        x = x.clone()
        for i in range(B):
            s = shifts[i].item()

            if nonmask[i].item():
                continue

            x[i] = torch.roll(x[i], int(shifts[i].item()), dims=-1)

            if s < 0:
                x[i, :, s:] = 0

            elif s > 0:
                x[i, :, :s] = 0

        return x


class AugmentAmplitudeShift(nn.Module):
    def __init__(self, max_shift_scale: float = 3, prob: float = 0.2):
        super().__init__()

        self.max_shift_scale = max_shift_scale
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        assert x.ndim == 3
        B, C, T = x.shape
        dvc = x.device

        max_shift = x.abs().mean().item() * self.max_shift_scale
        draws = torch.rand(B, C, 1, device=dvc)
        nonmask = torch.rand((B,), device=dvc) > self.prob

        shifts = max_shift * draws
        shifts = shifts * 2 - max_shift
        shifts[nonmask] = 0

        x = x + shifts

        return x


class AugmentZeroMasking(nn.Module):
    def __init__(self, max_mask_len: float = 0.2, prob: float = 0.2):
        super().__init__()

        self.max_mask_len = max_mask_len
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        assert x.ndim == 3
        B, C, T = x.shape

        x = x.clone()

        for i in range(B):
            for j in range(C):
                if torch.rand(1) < self.prob:
                    mask_len = int(torch.rand(1) * self.max_mask_len * T)
                    start = int(torch.rand(1) * (T - mask_len))
                    x[i, j, start : start + mask_len] = 0

        return x


class AugmentGaussianNoise(nn.Module):
    def __init__(self, std: float = 0.1, prob: float = 0.2):
        super().__init__()

        self.std = std
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        assert x.ndim == 3
        B, C, T = x.shape
        dvc = x.device

        nonmask = torch.rand(B, device=dvc) > self.prob
        noise_parts = torch.randn_like(x, device=dvc) * self.std
        noise_parts[nonmask] = 0

        return x + noise_parts


class AugmentBandStopFilter(nn.Module):
    def __init__(
        self,
        sample_rate: float = 200.0,
        prob: float = 0.2,
        min_freq: float = 2.8,
        max_freq: float = 82.5,
        width_hz: float = 5.0,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.prob = float(prob)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.width_hz = float(width_hz)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        assert x.ndim == 3
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        apply_mask = torch.rand(B, device=device) < self.prob

        # FFT frequency bins
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate).to(device)
        nyq = 0.5 * self.sample_rate

        min_f = max(0.0, self.min_freq)
        max_f = min(nyq, self.max_freq)
        if max_f <= min_f:
            return x

        # Fixed-width stopband
        half_w = 0.5 * self.width_hz
        center = min_f + (max_f - min_f) * torch.rand(B, device=device)
        low = torch.clamp(center - half_w, min=0.1)
        high = torch.clamp(center + half_w, max=nyq - 0.1)

        # Frequency mask
        f = freqs[None, :]  # (1, F)
        band = (f >= low[:, None]) & (f <= high[:, None])
        band = band & apply_mask[:, None]

        # 1 = pass, 0 = stop
        response = (~band).to(dtype=dtype)

        # Apply in frequency domain
        X = torch.fft.rfft(x, dim=-1)
        X = X * response[:, None, :].to(X.dtype)
        x_out = torch.fft.irfft(X, n=T, dim=-1)

        return x_out


class AugmentationPipeline(nn.Module):
    def __init__(self, augment_classes: list[str], kwarg_overrides: dict[str, dict[str, Any]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in augment_classes:
            if key not in AUGMENTATION_REGISTRY:
                raise ValueError(f"Unknown augmentation module: {key}")
            
        for key in kwarg_overrides:
            if key not in augment_classes:
                raise ValueError(f"Unknown augmentation module: {key}")

        self.aug_modules = nn.ModuleDict({key: AUGMENTATION_REGISTRY[key](**kwarg_overrides.get(key, {})) for key in augment_classes})

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        for module in self.aug_modules.values():
            x = module(x, *args, **kwargs)

        return x


AUGMENTATION_REGISTRY = {
    "amplitude_scale": AugmentAmplitudeScale,
    "time_shift": AugmentTimeShift,
    "amplitude_shift": AugmentAmplitudeShift,
    "zero_masking": AugmentZeroMasking,
    "gaussian_noise": AugmentGaussianNoise,
    "bandstop_filter": AugmentBandStopFilter,
}


class EEGAugmentationPipeline(AugmentationPipeline):
    def __init__(
        self,
        kwarg_overrides: dict[str, dict[str, Any]] = {},
        augment_classes=(
            "amplitude_scale",
            "time_shift",
            "amplitude_shift",
            "zero_masking",
            "gaussian_noise",
            "bandstop_filter",
        ),
        *args,
        **kwargs
    ):
        super().__init__(augment_classes, kwarg_overrides, *args, **kwargs)