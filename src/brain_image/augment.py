from __future__ import annotations
from abc import ABC, abstractmethod

from torch import Tensor
import torch
from torch import nn
from torchvision.transforms import v2 as tv2
from typing import Any, Sequence


class BaseAugment(nn.Module, ABC):
    def __init__(self, prob: float):
        super().__init__()

        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        if self.prob == 0:
            return x

        if self.prob == 1:
            return self.augment(x)

        x = x.clone()
        mask = torch.rand(x.shape[0], device=x.device) > self.prob

        if not mask.any().item():
            return x

        x[mask] = self.augment(x[mask])
        x = x.contiguous()
        return x

    @abstractmethod
    def augment(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class WrapAugment(BaseAugment):
    def __init__(self, augmentation: nn.Module, prob: float = 1):
        super().__init__(prob)
        self.augmentation = augmentation

    def augment(self, x):
        return self.augmentation(x)


class EEGAmplitudeScale(nn.Module):
    def __init__(self, min_scale: float = 0.5, max_scale: float = 2):
        super().__init__()

        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, x: Tensor) -> Tensor:
        # x: <B, C, T>
        assert x.ndim == 3
        B, C, T = x.shape
        dvc = x.device

        scale = (
            torch.rand((B, 1, 1), device=dvc) * (self.max_scale - self.min_scale)
            + self.min_scale
        )

        x = x * scale
        return x


class EEGTimeShift(nn.Module):
    def __init__(self, max_shift_scale: float = 0.2):
        super().__init__()

        self.max_shift_scale = max_shift_scale

    def forward(self, x: Tensor) -> Tensor:
        # x: <B, C, T>
        assert x.ndim == 3
        B, C, T = x.shape
        dvc = x.device

        # 1. Randomly select shifts for each sample
        # 2. Set shifts to 0 with probability 1 - prob
        # 3. Roll the T dimension by the shift for each sample

        shift_scale = int(T * self.max_shift_scale)

        shifts = torch.randint(
            low=-shift_scale,
            high=shift_scale,
            size=(B,),
            device=dvc,
        )

        for i in range(B):
            s = shifts[i].item()

            x[i] = torch.roll(x[i], int(shifts[i].item()), dims=-1)

            if s < 0:
                x[i, :, s:] = 0

            elif s > 0:
                x[i, :, :s] = 0

        return x


class EEGAmplitudeShift(nn.Module):
    def __init__(self, max_shift_scale: float = 3):
        super().__init__()

        self.max_shift_scale = max_shift_scale

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        assert x.ndim == 3
        B, C, T = x.shape
        dvc = x.device

        max_shift = x.abs().mean().item() * self.max_shift_scale
        draws = torch.rand(B, C, 1, device=dvc)

        shifts = max_shift * draws
        shifts = shifts * 2 - max_shift

        x = x + shifts

        return x


class EEGZeroMasking(nn.Module):
    def __init__(self, max_mask_len: float = 0.2):
        super().__init__()

        self.max_mask_len = max_mask_len

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        assert x.ndim == 3
        B, C, T = x.shape
        device = x.device

        max_len = max(1, int(self.max_mask_len * T))
        max_start = max(1, T - max_len + 1)

        mask_sizes = torch.randint(1, max_len + 1, (B, C), device=device)
        mask_starts = torch.randint(0, max_start, (B, C), device=device)
        mask_ends = mask_starts + mask_sizes  # <= T guaranteed by max_start construction

        t = torch.arange(T, device=device).view(1, 1, T)  # (1,1,T)
        mask = (t >= mask_starts.unsqueeze(-1)) & (t < mask_ends.unsqueeze(-1))  # (B,C,T)

        # In-place zeroing
        x.masked_fill_(mask, 0)

        return x

class EEGGaussianNoise(nn.Module):
    def __init__(self, std: float = 0.1):
        super().__init__()

        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        assert x.ndim == 3
        B, C, T = x.shape
        dvc = x.device

        noise_parts = torch.randn_like(x, device=dvc) * self.std

        return x + noise_parts


class EEGBandStopFilter(nn.Module):
    def __init__(
        self,
        sample_rate: float = 200.0,
        min_freq: float = 2.8,
        max_freq: float = 82.5,
        width_hz: float = 5.0,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.width_hz = float(width_hz)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        assert x.ndim == 3
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

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

        # 1 = pass, 0 = stop
        response = (~band).to(dtype=dtype)

        # Apply in frequency domain
        X = torch.fft.rfft(x, dim=-1)
        X = X * response[:, None, :].to(X.dtype)
        x_out = torch.fft.irfft(X, n=T, dim=-1)

        return x_out


class AugmentationPipeline(nn.Module):
    def __init__(self, augment_modules: list[BaseAugment], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.augment_modules = nn.ModuleList(augment_modules)

    def forward(self, x: Tensor, disabled: bool = False) -> Tensor:
        x = self.preprocess(x)

        if disabled:
            return x

        for augment in self.augment_modules:
            x = augment(x)
            
        x = self.postprocess(x)

        x = x.clone().contiguous()
        return x

    def preprocess(self, x: Tensor) -> Tensor:
        return x

    def postprocess(self, x: Tensor) -> Tensor:
        return x


AUGMENTATION_REGISTRY = {
    "amplitude_scale": EEGAmplitudeScale,
    "time_shift": EEGTimeShift,
    "amplitude_shift": EEGAmplitudeShift,
    "zero_masking": EEGZeroMasking,
    "gaussian_noise": EEGGaussianNoise,
    "bandstop_filter": EEGBandStopFilter,
}


class EEGAugmentationPipeline(AugmentationPipeline):
    # From M. Mohsenvand et al. "Contrastive Representation Learning for Electroencephalogram Classification"
    # Augmentations:
    # - Amplitude scale
    # - Time shift
    # - DC shift
    # - Zero masking
    # - Gaussian noise
    # - Band-Stop filtering
    def __init__(
        self,
        ampscale_min: float = 0.2,
        ampscale_max: float = 2.0,
        ampscale_prob: float = 0.2,
        timeshift_max_scale: float = 0.2,
        timeshift_prob: float = 0.2,
        ampshift_max_scale: float = 3.0,
        ampshift_prob: float = 0.2,
        zeromask_max_scale: float = 0.2,
        zeromask_prob: float = 0.2,
        blur_std: float = 0.2,
        blur_prob: float = 0.2,
        bandstop_sample_rate: int = 200,
        bandstop_min_freq: float = 2.8,
        bandstop_max_freq: float = 82.5,
        bandstop_width: float = 5,
        bandstop_prob: float = 0.2,
    ):
        super().__init__(
            [
                WrapAugment(
                    EEGAmplitudeScale(ampscale_min, ampscale_max),
                    ampscale_prob,
                ),
                WrapAugment(EEGTimeShift(timeshift_max_scale), timeshift_prob),
                WrapAugment(EEGAmplitudeShift(ampshift_max_scale), ampshift_prob),
                WrapAugment(EEGZeroMasking(zeromask_max_scale), zeromask_prob),
                WrapAugment(EEGGaussianNoise(blur_std), blur_prob),
                WrapAugment(
                    EEGBandStopFilter(
                        bandstop_sample_rate,
                        bandstop_min_freq,
                        bandstop_max_freq,
                        bandstop_width,
                    ),
                    bandstop_prob,
                ),
            ]
        )
        self._reshape_to_4d: bool = False
        self._stored_size: Sequence[int] | None = None

    def preprocess(self, x):
        if x.ndim == 4:
            self._reshape_to_4d = True
            self._stored_size = x.shape
            C, T = x.shape[-2], x.shape[-1]
            x = x.reshape(-1, C, T)

        return x

    def postprocess(self, x):
        if self._reshape_to_4d and x.ndim == 3:
            assert self._stored_size is not None
            x = x.reshape(self._stored_size)
            self._reshape_to_4d = False
            self._stored_size = None

        return x


class ImageAugmentationPipeline(AugmentationPipeline):
    # Augmentations:
    # - Random flips
    # - Random color jitter
    # - Gaussian blur

    def __init__(
        self,
        flip_prob: float = 0.5,
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.1,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.2,
        color_jitter_prob: float = 0.5,
        blur_kernel_size: int = 5,
        blur_prob: float = 0.1,
    ):
        super().__init__(
            [
                WrapAugment(tv2.RandomHorizontalFlip(flip_prob)),
                WrapAugment(
                    tv2.ColorJitter(
                        brightness=color_jitter_brightness,
                        contrast=color_jitter_contrast,
                        saturation=color_jitter_saturation,
                        hue=color_jitter_hue,
                    ),
                    color_jitter_prob,
                ),
                WrapAugment(
                    tv2.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.2, 2.0)),
                    blur_prob,
                ),
            ]
        )

    def preprocess(self, x: Tensor) -> Tensor:
        assert x.ndim == 4, "Expected input to be 4D (B, C, H, W)"

        if (x > 3).any():
            x = x / 255.0

        return x

    def postprocess(self, x: Tensor) -> Tensor:
        x = x.clamp(0, 1)
        return x


class LatentAffine(nn.Module):
    def __init__(
        self,
        scale_std: float = 0.1,
        shift_std: float = 0.1,
    ):
        super().__init__()
        self.scale_std = scale_std
        self.shift_std = shift_std

    def forward(self, x: Tensor) -> Tensor:
        # x: <B, D>
        # scale: B
        # Shift: B

        scale = torch.randn(x.shape[0], device=x.device) * self.scale_std + 1
        shift = torch.randn(x.shape[0], device=x.device) * self.shift_std

        return x * scale[:, None] + shift[:, None]


class LatentNoise(nn.Module):
    def __init__(
        self,
        noise_std: float = 0.05,
        relative: bool = True,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.relative = relative

    def forward(self, z: Tensor) -> Tensor:
        eps = torch.randn_like(z)

        if self.relative:
            scale = z.std(dim=-1, keepdim=True).clamp(min=1e-6)
            eps = eps * scale

        return z + eps * self.noise_std


class LatentDropout(nn.Module):
    def __init__(
        self,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, z: Tensor) -> Tensor:
        mask = torch.rand_like(z) > self.drop_prob
        return z * mask


class LatentAugmentationPipeline(AugmentationPipeline):
    def __init__(
        self, affine_scale_std: float = 0.1, affine_shift_std: float = 0.1, affine_prob: float = 0.5, noise_std: float = 0.05, noise_prob: float = 1.0, dropout_prob: float = 0.1,
    ):
        super().__init__([
            WrapAugment(LatentAffine(affine_scale_std, affine_shift_std), affine_prob),
            WrapAugment(LatentNoise(noise_std), noise_prob),
            WrapAugment(LatentDropout(dropout_prob), 1.0)
        ])
