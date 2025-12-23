from typing import Literal
import torch
import torch.nn as nn

import torchvision.transforms.v2 as tv2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from brain_image.model.img_encoder import DREAMSIM_IMAGE_ENCODER, DreamsimImageEncoder

class InfoNCELoss(nn.Module):
    def __init__(self, init_temperature: float = 0.07, max_scale: float = 100):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / init_temperature)))
        self.max_scale = max_scale
        self.loss_func = nn.CrossEntropyLoss()

    def forward(
        self, z_e: torch.Tensor, z_i: torch.Tensor, ignore_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z_i.size(0) != z_e.size(0):
            raise ValueError(f"z_e and z_i should have the same batch size, but got {z_e.size(0)} and {z_i.size(0)}")

        device = z_e.device
        logits = z_e @ z_i.T 
        logits_scaled = logits * self.logit_scale.exp().clamp(max=self.max_scale)
        if ignore_mask is not None:
            keep_mask = ~ignore_mask.to(device)
            logits_scaled = logits_scaled[keep_mask][:, keep_mask]

        labels = torch.arange(logits_scaled.size(0), device=device)
        loss = self.loss_func(
            logits_scaled, target=labels
        )

        return loss, logits

class CLIPLoss(nn.Module):
    def __init__(self, init_temperature: float = 0.07, max_scale: float = 100):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / init_temperature)))
        self.max_scale = max_scale
        self.loss_func = nn.CrossEntropyLoss()


    def forward(
        self, z_e: torch.Tensor, z_i: torch.Tensor, ignore_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z_i.size(0) != z_e.size(0):
            raise ValueError(f"z_e and z_i should have the same batch size, but got {z_e.size(0)} and {z_i.size(0)}")

        device = z_e.device
        logits = z_e @ z_i.T 
        logits_scaled = logits * self.logit_scale.exp().clamp(max=self.max_scale)
        if ignore_mask is not None:
            keep_mask = ~ignore_mask.to(device)
            logits_scaled = logits_scaled[keep_mask][:, keep_mask]

        labels = torch.arange(logits_scaled.size(0), device=device)
        loss_e = self.loss_func(
            logits_scaled, target=labels
        )
        loss_i = self.loss_func(
            logits_scaled.T, target=labels
        )
        loss = (loss_e + loss_i) * 0.5

        return loss, logits


class SigLipLoss(nn.Module):
    def __init__(self, init_temperature: float = 0.07, init_bias: float = 0.0, max_scale: float = 100, ignore_idx = -100):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / init_temperature)))
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))
        self.max_scale = max_scale
        self.ignore_idx = ignore_idx
        self.loss_func = nn.BCEWithLogitsLoss()


    def forward(
        self, z_e: torch.Tensor, z_i: torch.Tensor, ignore_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = z_e.device

        if z_i.size(0) != z_e.size(0):
            raise ValueError(f"z_e and z_i should have the same batch size, but got {z_e.size(0)} and {z_i.size(0)}")

        logits = z_e @ z_i.T 
        logits_scaled = logits * self.logit_scale.exp().clamp(max=self.max_scale) + self.logit_bias
        if ignore_mask is not None:
            keep_mask = ~ignore_mask.to(device)
            logits_scaled = logits_scaled[keep_mask][:, keep_mask]

        labels = torch.eye(logits_scaled.size(0), device=device) 
        #loss = self.loss_func(logits_scaled, labels)
        loss = nn.functional.binary_cross_entropy_with_logits(logits_scaled, labels)

        return loss, logits



class DreamsimLoss(nn.Module):
    def __init__(self, model_name: DREAMSIM_IMAGE_ENCODER = "synclr_vitb16", rescale_cutoff: float = 10, *args, **kwargs):
        super().__init__()
        self.dreamsim = DreamsimImageEncoder(model_name=model_name)

        self.processor = tv2.Compose([
            tv2.Resize(224, interpolation=tv2.InterpolationMode.BICUBIC, antialias=True),
            tv2.ToDtype(torch.float32),
        ])
        self.rescale_cutoff = rescale_cutoff

    @torch.compiler.disable()
    def _prep_latent(self, x):
        x = self.processor(x)
        if x.max() > self.rescale_cutoff:
            x = x / 255.0
        return x

    def forward(self, pred, gt):
        pred = self._prep_latent(pred)
        gt = self._prep_latent(gt)

        cos = self.dreamsim.model(pred, gt)
        return cos.mean()




class LPIPSLoss(torch.nn.Module):
    def __init__(self, net_type: Literal["alex", "vgg", "squeeze"] = "vgg", normalize: bool = True, rescale_cutoff: float = 10, *args, **kwargs):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=normalize)

        self.processor = tv2.Compose([
            tv2.Resize(224, interpolation=tv2.InterpolationMode.BICUBIC, antialias=True),
            tv2.ToDtype(torch.float32),
        ])
        self.rescale_cutoff = rescale_cutoff


    @torch.compiler.disable()
    def _prep_latent(self, x):
        x = self.processor(x)
        if x.max() > self.rescale_cutoff:
            x = x / 255.0
        x = torch.clamp(x, 0, 1)
        return x


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._prep_latent(x)        
        y = self._prep_latent(y)

        return self.lpips(x, y).mean()