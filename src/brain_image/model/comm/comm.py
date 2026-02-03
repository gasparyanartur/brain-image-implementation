# From https://github.com/Duplums/CoMM/blob/main/pl_modules/comm.py

from functools import lru_cache
from torch import nn
import torch
from collections import OrderedDict
from typing import Dict, List, Union

from brain_image.model.comm.base import BaseModel
from brain_image.model.comm.mmfusion import MMFusion
from brain_image.model.comm.comm_loss import CoMMLoss


@lru_cache(maxsize=1)
@torch.no_grad()
def gen_all_possible_masks(n_mod: int):
    """
    :param n_mod: int
    :return: a list of `n_mod` + 1 boolean masks [Mi] such that all but one bool are False.
        A last bool mask is added where all bool are True
    Examples:
    *   For n_mod==2:
        masks == [[True, False], [False, True], [True, True]]
    *   For n_mod == 3:
        masks == [[True, False, False], [False, True, False], [False, False, True], [True, True, True]]
    """
    masks = []
    for L in range(n_mod):
        mask = [s == L for s in range(n_mod)]
        masks.append(mask)
    masks.append([True for _ in range(n_mod)])
    return masks


class CoMM(BaseModel):
    """Contrastive MultiModal learning allowing the communication between modalities
    in a single multimodal space [1].

    It encodes a pair of mulitmodal data and outputs a pair of representations through
    a single multimodal encoder.

    [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(
        self,
        encoder: MMFusion,
        projection: nn.Module,
    ):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
        """
        super(CoMM, self).__init__()

        # create the encoder
        self.encoder = encoder

        # build a 3-layers projector
        self.head = projection

    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(
            OrderedDict(
                [
                    ("layer1", nn.Linear(in_dim, mlp_dim)),
                    ("bn1", nn.SyncBatchNorm(mlp_dim)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("layer2", nn.Linear(mlp_dim, mlp_dim)),
                    ("bn2", nn.SyncBatchNorm(mlp_dim)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("layer3", nn.Linear(mlp_dim, out_dim)),
                ]
            )
        )

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x1, x2: list of tensors of shape (B, C, H, W) for each modality
        # x1, x2: (mod1_x, mod2_x)
        # x1: (aug1(mod1_x1), aug2(mod2_x1))
        # x2: (aug1(mod1_x2), aug2(mod2_x2))

        # compute features for all modalities
        all_masks = gen_all_possible_masks(len(x1))
        z1 = self.encoder(x1, mask_modalities=all_masks)
        z2 = self.encoder(x2, mask_modalities=all_masks)
        z1 = [self.head.forward(z) for z in z1]
        z2 = [self.head.forward(z) for z in z2]
        return {"aug1_embed": z1, "aug2_embed": z2, "prototype": -1} # type: ignore
    
    def encode_token(self, z: List[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        # x: (mod1_x, mod2_x)
        if isinstance(z, torch.Tensor):
            z = [z]
         
        z = self.encoder.fusion_transformer.forward(z)  # type: ignore
        z = self.head.forward(z)


        return z # type: ignore

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
        Extract multimodal features from the encoder.
        Args:
             loader: Dataset loader to serve `(X, y)` tuples.
             kwargs: given to `encoder.forward()`
        Returns:
             Pair (Z,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor):  # needs to cast it as list of one modality
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode():
                # compute output
                output = self.encoder(X_, **kwargs)
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(
            self.device
        )

    def encode_feature(self, x: torch.Tensor | List[torch.Tensor], mod_idx: int | List[int], head: bool = True) -> torch.Tensor:
        # X is a list of each modality
        if isinstance(x, torch.Tensor):
            x = [x]

        if isinstance(mod_idx, int):
            mod_idx = [mod_idx]

        assert len(x) == len(mod_idx)

        z = self.encoder(x, mod_idx=mod_idx)
        if head:
            z = self.head(z)
        return z
