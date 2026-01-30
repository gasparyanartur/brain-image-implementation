# From https://github.com/Duplums/CoMM/blob/main/pl_modules/base.py

from pytorch_lightning import LightningModule
from torch import Tensor
from typing import Tuple, Dict
from abc import ABC, abstractmethod
import torch
import math
import sys
from brain_image.model.comm.utils import set_weight_decay_per_param, LinearWarmupCosineAnnealingLR


class BaseModel(ABC, LightningModule):
    """
        Base model for Self-Supervised Learning (SSL), Vision-Language (VL) or Language-Guided (LG) models.
        We expect any `BaseModel` to implement a features extractor.
    """

    def __init__(self, optim_kwargs: Dict = {}):
        super().__init__()
        self.optim_kwargs = optim_kwargs


    @abstractmethod
    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs) \
            -> Tuple[Tensor, Tensor]:
        """
        Extract global average pooled visual features.
        Args:
            loader: Dataset loader to serve ``(image, label)`` tuples.
        Returns:
            Pair (X,y) corresponding to extracted features and corresponding labels
        """
        pass
