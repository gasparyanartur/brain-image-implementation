import torch
import torch.nn as nn


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

        if z_e.size(0) != z_i.size(0):
            raise ValueError(f"z_e and z_i should have the same batch size, but got {z_e.size(0)} and {z_i.size(0)}")

        if labels is None:
            labels = torch.ones(z_i.size(0), device=device, dtype=torch.float).diag()

        if labels.size(0) != z_i.size(0) or labels.size(1) != z_i.size(0):
            raise ValueError(f"Labels shape should be ({z_i.size(0)}, {z_i.size(0)}), but got {labels.shape}")

        logits = z_e @ z_i.T 
        logits_scaled = logits * self.logit_scale.exp().clamp(max=self.max_scale)
        if ignore_mask is not None:
            labels[ignore_mask.to(device)] = self.ignore_idx

        loss_e = self.loss_func(
            logits_scaled, labels
        )
        loss_i = self.loss_func(
            logits_scaled.T, labels
        )
        loss = (loss_e + loss_i) * 0.5

        return loss, logits
