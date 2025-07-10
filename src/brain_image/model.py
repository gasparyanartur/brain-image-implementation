from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, field_validator
import torch
import torch.nn as nn
import einops
import math
import itertools as it
from lightning import LightningModule

from brain_image.configs import BaseConfig
from brain_image.data import EEGDataModule, EEGDatasetConfig
import dreamsim
from dreamsim.feature_extraction.load_synclr_as_dino import load_synclr_as_dino
from dreamsim.feature_extraction.vision_transformer import VisionTransformer


def load_image_encoder(model_name: str, models_path: Path) -> VisionTransformer:
    try:
        logging.info(f"Loading {model_name} model...")
        match model_name:
            case "synclr":
                model = load_synclr_as_dino(16, load_dir=str(models_path))
            case "aligned_synclr":
                dreamsim_model, _ = dreamsim.dreamsim(
                    dreamsim_type="synclr_vitb16", cache_dir=str(models_path)
                )
                model = dreamsim_model.base_model.model.extractor_list[0].model  # type: ignore
            case _:
                raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        logging.error(f"Error loading {model_name} model: {e}")
        raise e
    else:
        logging.info(f"Model {model_name} loaded successfully.")

    return model


class ModelConfig(BaseConfig):
    def create_model(self) -> Model:
        raise NotImplementedError


class Model(LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config


class EEGEncoderConfig(ModelConfig):
    embed_dim: int = 40

    temporal_kernel_size: int = 25
    spatial_kernel_size: int = 17
    temporal_pool_size: int = 41
    temporal_stride: int = 1
    hidden_dim: int = 40
    dropout: float = 0.5
    final_spatiotemporal_size: int = 36

    @property
    def encoded_dim(self) -> int:
        return self.embed_dim * self.final_spatiotemporal_size

    def create_model(self) -> EEGEncoder:
        return EEGEncoder(self)


class DebugLayer(nn.Module):
    def __init__(self, note: str = ""):
        super(DebugLayer, self).__init__()
        self.note = note

    def forward(self, x):
        print(f"(debug): {x.shape} - {self.note}")
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        temporal_pool_size: int,
        temporal_stride: int,
        hidden_dim: int,
        final_spatiotemporal_size: int,
        dropout: float = 0.5,
    ):
        # Adapted from https://github.com/eeyhsong/NICE-EEG
        super().__init__()

        self.spatiotemporal_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(1, temporal_kernel_size)),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=(1, temporal_pool_size),
                stride=(1, temporal_stride),
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(spatial_kernel_size, 1), bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

        self.projection = nn.Conv2d(hidden_dim, embed_dim, kernel_size=(1, 1))
        self.final_proj = nn.AdaptiveAvgPool1d((final_spatiotemporal_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b s t -> b 1 s t")
        x = self.spatiotemporal_conv(x)
        x = self.projection(x)
        x = einops.rearrange(x, "b e (s) (t) -> b e (s t)")
        x = self.final_proj(x)
        x = einops.rearrange(x, "b e st -> b st e")
        return x

    def jit_compile(self):
        # Compile the model for faster inference
        self.spatiotemporal_conv = torch.jit.script(self.spatiotemporal_conv)
        self.projection = torch.jit.script(self.projection)
        return self


class EEGEncoder(Model):
    def __init__(
        self,
        config: EEGEncoderConfig = EEGEncoderConfig(),
    ):
        # Adapted from https://github.com/eeyhsong/NICE-EEG
        super(EEGEncoder, self).__init__(config)

        self.patch_embedding = PatchEmbedding(
            embed_dim=config.embed_dim,
            temporal_kernel_size=config.temporal_kernel_size,
            spatial_kernel_size=config.spatial_kernel_size,
            temporal_pool_size=config.temporal_pool_size,
            temporal_stride=config.temporal_stride,
            hidden_dim=config.hidden_dim,
            final_spatiotemporal_size=config.final_spatiotemporal_size,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = x.flatten(start_dim=1)

        return x

    def jit_compile(self):
        self.patch_embedding = self.patch_embedding.jit_compile()
        return self


class LatentProjector(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1440,
        proj_dim: int = 768,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.l_proj = nn.Linear(embed_dim, proj_dim)
        self.l_inner = nn.Sequential(
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x = self.l_proj(x)
        x = self.l_inner(x) + x_res
        x = self.norm(x)

        return x


class NICEConfig(ModelConfig):
    eeg_config: EEGEncoderConfig = EEGEncoderConfig()
    model_name: Literal[
        "synclr",
        "aligned_synclr",
    ]
    project_dim: int = 256
    img_latent_dim: int = 768
    encoder_lr: float = 8e-3
    projector_lr: float = 8e-3
    lr_scheduler: Literal["none", "cosine_anneal"] = "cosine_anneal"
    betas: tuple[float, float] = (0.9, 0.999)
    encoder_min_lr: float = 1e-4
    projector_min_lr: float = 1e-4
    projector_warmup_epochs: int = 2
    encoder_warmup_epochs: int = 4
    warmup_start_frac: float = 0.1
    max_epochs: int = 100
    num_workers: int = 8
    temperature_init: float = math.log(1 / 0.07)
    data_seed: int = 42

    @field_validator("eeg_config", mode="before")
    @classmethod
    def validate_eeg_config(cls, v):
        """Convert dict to EEGEncoderConfig if needed."""
        if isinstance(v, dict):
            return EEGEncoderConfig.model_validate(v)
        return v


class NICEModel(Model):
    def __init__(
        self,
        config: NICEConfig | dict[str, Any],
        dataset_config: EEGDatasetConfig | dict[str, Any] = EEGDatasetConfig(),
        compile: bool = True,
        init_weights: bool = True,
    ):
        # Convert dicts to Pydantic models if they aren't already
        if isinstance(config, dict):
            config = NICEConfig.model_validate(config)

        if isinstance(dataset_config, dict):
            dataset_config = EEGDatasetConfig.model_validate(dataset_config)

        # Recursively convert all dicts to NICEConfig
        super(NICEModel, self).__init__(config)

        self.automatic_optimization = False
        self.config = config
        self.eeg_encoder = EEGEncoder(config.eeg_config)
        self.eeg_projector = LatentProjector(
            embed_dim=config.eeg_config.encoded_dim,
            proj_dim=config.project_dim,
        )
        self.img_projector = LatentProjector(
            embed_dim=config.img_latent_dim,
            proj_dim=config.project_dim,
        )
        self.temperature = nn.Parameter(
            torch.tensor(config.temperature_init, dtype=torch.float32)
        )
        self.loss = nn.CrossEntropyLoss()

        if init_weights:
            self._init_normal_weights()

        # Create the data module
        self.data_module = EEGDataModule(dataset_config, model_name=config.model_name)

        if compile:
            logging.info("Compiling model...")
            self.eeg_encoder = torch.compile(self.eeg_encoder)
            self.eeg_projector = torch.compile(self.eeg_projector)
            self.img_projector = torch.compile(self.img_projector)

        self.save_hyperparameters(
            {
                "config": config.model_dump(mode="json"),
                "dataset_config": dataset_config.model_dump(mode="json"),
            },
        )

    def _init_normal_weights(self):
        """Initialize weights for the model."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        """Configure optimizers for the model."""

        encoder_optimizer = torch.optim.Adam(
            self.eeg_encoder.parameters(),
            lr=self.config.encoder_lr,
            betas=self.config.betas,
        )
        projector_optimizer = torch.optim.Adam(
            [
                {
                    "params": self.eeg_projector.parameters(),
                    "lr": self.config.projector_lr,
                },
                {
                    "params": self.img_projector.parameters(),
                    "lr": self.config.projector_lr,
                },
                {
                    "params": [self.temperature],
                    "lr": self.config.projector_lr,
                },
            ],
            betas=self.config.betas,
        )

        encoder_schedulers = []
        projector_schedulers = []
        projector_milestones = []
        encoder_milestones = []
        if self.config.encoder_warmup_epochs > 0:
            encoder_schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    encoder_optimizer,
                    start_factor=self.config.warmup_start_frac,
                    total_iters=self.config.encoder_warmup_epochs,
                )
            )
            encoder_milestones.append(self.config.encoder_warmup_epochs)

        if self.config.projector_warmup_epochs > 0:
            projector_schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    projector_optimizer,
                    start_factor=self.config.warmup_start_frac,
                    total_iters=self.config.projector_warmup_epochs,
                )
            )
            projector_milestones.append(self.config.projector_warmup_epochs)

        match self.config.lr_scheduler:
            case "none":
                encoder_schedulers.append(
                    torch.optim.lr_scheduler.ConstantLR(
                        encoder_optimizer,
                        factor=1.0,
                    )
                )
                projector_schedulers.append(
                    torch.optim.lr_scheduler.ConstantLR(
                        projector_optimizer,
                        factor=1.0,
                    )
                )
            case "cosine_anneal":
                encoder_schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        encoder_optimizer,
                        T_max=self.config.max_epochs,
                        eta_min=self.config.encoder_min_lr,
                    )
                )
                projector_schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        projector_optimizer,
                        T_max=self.config.max_epochs,
                        eta_min=self.config.projector_min_lr,
                    )
                )
            case _:
                raise ValueError(f"Unknown lr_scheduler: {self.config.lr_scheduler}")

        encoder_scheduler = torch.optim.lr_scheduler.SequentialLR(
            encoder_optimizer,
            schedulers=encoder_schedulers,
            milestones=encoder_milestones,
        )
        projector_scheduler = torch.optim.lr_scheduler.SequentialLR(
            projector_optimizer,
            schedulers=projector_schedulers,
            milestones=projector_milestones,
        )
        return [
            {
                "optimizer": encoder_optimizer,
                "lr_scheduler": encoder_scheduler,
                "interval": "step",
                "frequency": 1,
            },
            {
                "optimizer": projector_optimizer,
                "lr_scheduler": projector_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        ]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the training dataloader."""
        return self.data_module.train_dataloader()

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the validation dataloader."""
        return self.data_module.val_dataloader()

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the test dataloader."""
        return self.data_module.test_dataloader()

    @property
    @lru_cache(maxsize=1)
    def num_train_batches(self) -> int:
        """Return the length of the training dataloader."""
        return len(self.data_module.train_dataloader())

    def forward(self, img_latent: torch.Tensor, eeg_data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        eeg_latent = self.eeg_encoder(eeg_data)
        eeg_latent = self.eeg_projector(eeg_latent)
        img_latent = self.img_projector(img_latent)

        sim = compute_similarity(
            eeg_latent=eeg_latent,
            img_latent=img_latent,
            temperature=self.temperature,
        )

        return sim

    def get_loss(self, sim: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        loss = compute_cross_entropy_loss(sim)
        return loss

    def get_top_n_accuracy(self, sim: torch.Tensor, n: int = 1) -> float:
        """Compute top-n accuracy."""
        labels = torch.arange(sim.size(0), device=sim.device)
        # Ensure n doesn't exceed batch size
        n = min(n, sim.size(0))
        top_n = sim.topk(n, dim=-1).indices

        correct = top_n == labels.unsqueeze(1)
        return (correct.any(dim=-1).float().sum() / correct.size(0)).item()

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        """Training step for the model."""
        img_latent = batch["img_latent"].to(self.device, dtype=self.dtype)
        eeg_data = batch["eeg_data"].to(self.device, dtype=self.dtype)

        sim = self(img_latent, eeg_data)
        loss = self.get_loss(sim)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        for opt in optimizers:
            opt.zero_grad()

        self.manual_backward(loss)

        for opt in optimizers:
            opt.step()

        # Step the schedulers on epoch end
        if batch_idx == self.num_train_batches - 1:
            for scheduler in schedulers:
                if scheduler is None:
                    continue

                scheduler.step()  # type: ignore

        return loss

    def validation_step(self, batch, batch_idx):
        img_latent = batch["img_latent"].to(self.device, dtype=self.dtype)
        eeg_data = batch["eeg_data"].to(self.device, dtype=self.dtype)

        with torch.no_grad():
            sim = self(img_latent, eeg_data)
            loss = self.get_loss(sim)

            top1_acc = self.get_top_n_accuracy(sim, n=1)
            top3_acc = self.get_top_n_accuracy(sim, n=3)
            top5_acc = self.get_top_n_accuracy(sim, n=5)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/top1_acc", top1_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/top3_acc", top3_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/top5_acc", top5_acc, prog_bar=False, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "top5_acc": top5_acc,
        }

    def test_step(self, batch, batch_idx):
        img_latent = batch["img_latent"].to(self.device, dtype=self.dtype)
        eeg_data = batch["eeg_data"].to(self.device, dtype=self.dtype)

        with torch.no_grad():
            sim = self(img_latent, eeg_data)
            loss = self.get_loss(sim)
            top1_acc = self.get_top_n_accuracy(sim, n=1)
            top3_acc = self.get_top_n_accuracy(sim, n=3)
            top5_acc = self.get_top_n_accuracy(sim, n=5)

        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/top1_acc", top1_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test/top3_acc", top3_acc, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log(
            "test/top5_acc", top5_acc, prog_bar=False, on_step=False, on_epoch=True
        )
        return {
            "loss": loss,
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "top5_acc": top5_acc,
        }


@torch.jit.script
def compute_cross_entropy_loss(sim: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss."""
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_e = nn.functional.cross_entropy(sim, labels)
    loss_i = nn.functional.cross_entropy(sim.T, labels)
    loss = (loss_e + loss_i) / 2
    return loss


@torch.jit.script
def compute_similarity(
    eeg_latent: torch.Tensor,
    img_latent: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Compute similarity between EEG and image latents."""
    eeg_latent = nn.functional.normalize(eeg_latent, dim=-1)
    img_latent = nn.functional.normalize(img_latent, dim=-1)
    sim = (eeg_latent @ img_latent.T) * torch.exp(temperature)
    return sim
