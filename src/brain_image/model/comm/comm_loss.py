# From https://github.com/Duplums/CoMM/blob/main/losses/comm_loss.py

import torch.nn.functional as func
import torch
import torch.nn as nn
from brain_image.model.comm.utils import all_gather_batch_with_grad


class CoMMLoss(nn.Module):
    """
        Normalized Temperature Cross-Entropy Loss for Multi-Modal Contrastive Learning as defined in CoMM [1]

        [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self, temperature=0.1, weights=None, skip_idxs: list[int] = []):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature)) 
        self.weights = weights
        self.skip_idxs = skip_idxs
        self.INF = 1e8

    def infonce(self, z1, z2):
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        
        return loss

    def forward(self, z1: list[torch.Tensor], z2: list[torch.Tensor], prototype_idx: int, norm: bool = True):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z 
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        assert len(z1) == len(z2)
        n_emb = len(z1)

        # Apply InfoNCE between a "prototype embedding" and all the others
        losses = {}

        zp1 = z1[prototype_idx]
        zp2 = z2[prototype_idx]

        if norm:
            zp1 = func.normalize(zp1, p=2, dim=-1)
            zp2 = func.normalize(zp2, p=2, dim=-1)

        for m in range(n_emb):
            if m in self.skip_idxs:
                continue

            zm1 = z1[m]
            zm2 = z2[m]

            if norm and m != prototype_idx:
                zm1 = func.normalize(zm1, p=2, dim=-1)
                zm2 = func.normalize(zm2, p=2, dim=-1)

            loss1 = self.infonce(zm1, zp2)
            loss2 = self.infonce(zm2, zp1)
            
            loss = ((loss1 + loss2) / 2.)
            losses[m] = loss
        
        return losses

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)