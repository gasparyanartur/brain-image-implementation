import torch
import torch.nn as nn

import torchvision.transforms.v2 as tv2
from brain_image.model.img_encoder import DREAMSIM_IMAGE_ENCODER, DreamsimImageEncoder


class CLIPLoss(nn.Module):
    def __init__(self, init_temperature: float = 0.07, max_scale: float = 100, ignore_idx = -100):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / init_temperature)))
        self.max_scale = max_scale
        self.ignore_idx = ignore_idx
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)


    def forward(
        self, z_e: torch.Tensor, z_i: torch.Tensor, labels: torch.Tensor | None = None, ignore_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = z_e.device

        B = z_e.size(0)

        if z_i.size(0) != B:
            raise ValueError(f"z_e and z_i should have the same batch size, but got {B} and {z_i.size(0)}")

        logits = z_e @ z_i.T 
        logits_scaled = logits * self.logit_scale.exp().clamp(max=self.max_scale)
        if ignore_mask is not None:
            keep_mask = ~ignore_mask.to(device)
            logits_scaled = logits_scaled[keep_mask][:, keep_mask]
            B = logits_scaled.size(0)

        if labels is None:
            labels = torch.arange(B, device=device)

        if labels.ndim != 1 or labels.size(0) != B:
            raise ValueError(f"Labels shape should be ({B},), but got {labels.shape}")

        loss_e = self.loss_func(
            logits_scaled, target=labels
        )
        loss_i = self.loss_func(
            logits_scaled.T, target=labels
        )
        loss = (loss_e + loss_i) * 0.5

        return loss, logits



class DreamsimLoss(nn.Module):
    def __init__(self, model_name: DREAMSIM_IMAGE_ENCODER = "synclr_vitb16", rescale_cutoff: float = 10, *args, **kwargs):
        super().__init__()
        self.dreamsim = DreamsimImageEncoder(model_name=model_name)
        self.dreamsim.eval()

        self.processor = tv2.Compose([
            tv2.ToImage(),
            tv2.Resize(224, interpolation=tv2.InterpolationMode.BICUBIC, antialias=True),
            tv2.ToDtype(torch.float32),
        ])
        self.rescale_cutoff = rescale_cutoff

    def forward(self, pred, gt):
        pred = self.processor(pred)
        gt = self.processor(gt)

        if pred.max() > self.rescale_cutoff:
            pred = pred / 255.0
        if gt.max() > self.rescale_cutoff:
            gt = gt / 255.0

        cos = self.dreamsim.model(pred, gt)
        return cos.mean()