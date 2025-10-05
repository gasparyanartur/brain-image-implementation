import torch
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self, init_temperature: float = 0.07, max_scale: float = 100):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / init_temperature)))
        self.max_scale = max_scale

    def forward(
        self, z_e: torch.Tensor, z_i: torch.Tensor, labels: torch.Tensor | None = None, symmetric: bool = True, reduce: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels is None:
            labels = torch.arange(z_i.size(0), device=z_i.device).long()

        sim = z_e @ z_i.T 
        sim_scaled = sim * self.logit_scale.exp().clamp(self.max_scale)

        loss_e = torch.nn.functional.cross_entropy(
            sim_scaled, labels, reduction = "mean" if reduce else "none"
        )
        if not symmetric:
            return loss_e, sim

        loss_i = torch.nn.functional.cross_entropy(
            sim_scaled.T, labels, reduction = "mean" if reduce else "none"
        )
        loss = (loss_e + loss_i) * 0.5

        return loss, sim


class InfoNCELoss(nn.Module):
    def __init__(self, init_temperature: float = 0.04, max_scale: float = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / init_temperature)))
        self.max_scale = max_scale

    @staticmethod
    def _get_directional_loss(sim: torch.Tensor, neg_mask: torch.Tensor, reduce: bool = True):
        row_max = sim.max(dim=-1, keepdim=True).values
        logits_row = sim - row_max

        row_log_denom = torch.logsumexp(logits_row, dim=-1)

        row_correct = logits_row.masked_fill(neg_mask, -torch.inf)
        row_log_numer = torch.logsumexp(row_correct, dim=-1)

        row_loss = -(row_log_numer - row_log_denom)
        if reduce:
            row_loss = row_loss.mean()
        return row_loss

    def forward(self, z_e: torch.Tensor, z_i: torch.Tensor, labels: torch.Tensor, symmetric: bool = True, reduce: bool = True):
        neg_mask = ~labels
        sim = z_e @ z_i.T
        scale = self.logit_scale.exp().clamp(self.max_scale)
        logits = sim * scale

        loss_e = self._get_directional_loss(logits, neg_mask, reduce=reduce)

        if not symmetric:
            return loss_e, sim

        loss_i = self._get_directional_loss(logits.T, neg_mask)
        loss = 0.5 * (loss_e + loss_i)

        return loss, sim