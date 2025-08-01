from __future__ import annotations

from collections import OrderedDict
from functools import cached_property, lru_cache
import logging
from pathlib import Path
import re
import tempfile
from typing import Any, Literal
from pydantic import BaseModel, field_validator
import torch
import torch.nn as nn
import einops
import math
import itertools as it
from lightning import LightningModule

from brain_image.configs import BaseConfig, get_device, get_device_str
from brain_image.data import EEGDataModule, EEGDatasetConfig
import dreamsim
from dreamsim.model import PerceptualModel


model_name_options = ["synclr", "clip"]
patch_size_options = ["16", "32"]
aligned_options = ["aligned", "unaligned"]
normalize_options = ["norm", "unnorm"]

default_aligned_option = "aligned"
default_patch_size = "16"
default_normalize_option = "unnorm"


def extract_model_config(model_config_str: str) -> tuple[str, str, str, str]:
    # Extracts whether aligned/unaligned, model_name, and patch_size (16 or 32)
    name_parts = model_config_str.split("_")
    if len(name_parts) == 1:
        aligned_option, model_name, patch_size, normalize_option = (
            default_aligned_option,
            name_parts[0],
            default_patch_size,
            default_normalize_option,
        )
    elif len(name_parts) == 2:
        aligned_option, model_name, patch_size, normalize_option = (
            default_aligned_option,
            name_parts[0],
            name_parts[1],
            default_normalize_option,
        )
    elif len(name_parts) == 3:
        aligned_option, model_name, patch_size, normalize_option = (
            name_parts[0],
            name_parts[1],
            name_parts[2],
            default_normalize_option,
        )
    elif len(name_parts) == 4:
        aligned_option, model_name, patch_size, normalize_option = (
            name_parts[0],
            name_parts[1],
            name_parts[2],
            name_parts[3],
        )
    else:
        raise ValueError(f"Invalid model name: {model_config_str}")

    if aligned_option not in aligned_options:
        raise ValueError(f"Invalid aligned option: {aligned_option}")
    if model_name not in model_name_options:
        raise ValueError(f"Invalid model name: {model_name}")
    if patch_size not in patch_size_options:
        raise ValueError(f"Invalid patch size: {patch_size}")
    if normalize_option not in normalize_options:
        raise ValueError(f"Invalid normalize option: {normalize_option}")

    return aligned_option, model_name, patch_size, normalize_option


def load_image_encoder(
    model_config_str: str,
    models_path: Path,
    download_weights: bool = True,
    device: str | None = None,
) -> PerceptualModel:
    try:
        aligned_option, model_name, patch_size, normalize_option = extract_model_config(
            model_config_str
        )
        logging.info(
            f"Loading {model_name} model with {patch_size} patch size and {aligned_option} alignment and {normalize_option} normalization..."
        )

        model_type = f"{model_name}_vitb{patch_size}"
        if download_weights:
            dreamsim.model.download_weights(
                dreamsim_type=model_type,
                cache_dir=str(models_path),
            )

        if aligned_option == "unaligned":
            model = PerceptualModel(
                model_type=model_type,
                normalize_embeds=normalize_option == "norm",
                stride=patch_size,
                load_dir=str(models_path),
                baseline=True,
                device=device or get_device_str(),
            )

        else:
            model, _ = dreamsim.dreamsim(
                dreamsim_type=model_type,
                cache_dir=str(models_path),
                normalize_embeds=normalize_option == "norm",
                device=device or get_device_str(),
            )

    except Exception as e:
        logging.error(f"Error loading {model_name} model: {e}")
        raise e
    else:
        logging.info(f"Model {model_name} loaded successfully.")

    return model  # type: ignore


class ModelConfig(BaseConfig):
    def create_model(self) -> Model:
        raise NotImplementedError


class Model(LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config


class EEGEncoderConfig(ModelConfig):
    f1: int = 64
    f2: int = 64
    pool1: int = 2
    stride1: int = 2
    pool2: int = 1
    stride2: int = 1
    kernel1: int = 29
    kernel2: int = 17
    dropout: float = 0.5
    embed_dim: int = 40

    def create_model(self) -> EEGEncoder:
        return EEGEncoder(self)


class DebugLayer(nn.Module):
    def __init__(self, note: str = ""):
        super(DebugLayer, self).__init__()
        self.note = note

    def forward(self, x):
        print(f"(debug): {x.shape} - {self.note}")
        return x


class WrapDebugSequential(nn.Module):
    def __init__(self, seq: nn.Sequential, note: str = ""):
        super(WrapDebugSequential, self).__init__()
        self.seq = seq

        self.debug_layers = nn.ModuleList(
            [DebugLayer(f"layer_{i}") for i in range(len(seq))]
        )

    def forward(self, x):
        for i, layer in enumerate(self.seq):
            x = layer(x)
            print(f"(debug): {i} - {x.shape} - {self.debug_layers[i].note}")
        return x


class EEGEncoder(Model):
    def __init__(
        self,
        config: EEGEncoderConfig = EEGEncoderConfig(),
    ):
        # Adapted from https://github.com/eeyhsong/NICE-EEG
        super(EEGEncoder, self).__init__(config)

        self.patch_embedding = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            1,
                            config.f1,
                            kernel_size=(1, config.kernel1),
                            stride=(1, 1),
                            bias=False,
                        ),
                    ),
                    ("bn1", nn.BatchNorm2d(config.f1)),
                    ("elu1", nn.ELU()),
                    ("dropout1", nn.Dropout(config.dropout, inplace=True)),
                    (
                        "pool1",
                        nn.AvgPool2d((1, config.pool1), stride=(1, config.stride1)),
                    ),
                    (
                        "conv2",
                        nn.Conv2d(
                            config.f1,
                            config.f2,
                            kernel_size=(config.kernel2, 1),
                            stride=(1, 1),
                            bias=False,
                        ),
                    ),
                    ("bn2", nn.BatchNorm2d(config.f2)),
                    ("elu2", nn.ELU()),
                    ("dropout2", nn.Dropout(config.dropout, inplace=True)),
                    (
                        "pool2",
                        nn.AvgPool2d((1, config.pool2), stride=(1, config.stride2)),
                    ),
                    (
                        "projection",
                        nn.Conv2d(config.f2, config.embed_dim, kernel_size=(1, 1)),
                    ),
                ]
            ),
        )
        # self.patch_embedding = WrapDebugSequential(self.patch_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b s t -> b 1 s t")
        x = self.patch_embedding(x)
        x = einops.rearrange(x, "b e (s) (t) -> b (s t e)")
        return x


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
    model_name: str = "aligned_synclr_16"
    project_dim: int = 256
    eeg_latent_dim: int = 1440
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
    project_image: bool = False

    @cached_property
    def latent_config(self) -> tuple[str, str, str, str]:
        return extract_model_config(self.model_name)

    @cached_property
    def embed_name(self) -> str:
        _, model_name, _, _ = self.latent_config
        return model_name

    @cached_property
    def embed_aligned_option(self) -> str:
        aligned_option, _, _, _ = self.latent_config
        return aligned_option

    @cached_property
    def embed_patch_size(self) -> str:
        _, _, patch_size, _ = self.latent_config
        return patch_size

    @cached_property
    def embed_normalized(self) -> bool:
        _, _, _, normalize_option = self.latent_config
        return normalize_option == "norm"

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
            embed_dim=config.eeg_latent_dim,
            proj_dim=config.project_dim,
        )
        self.img_projector = (
            LatentProjector(
                embed_dim=config.img_latent_dim,
                proj_dim=config.project_dim,
            )
            if config.project_image
            else None
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
            if self.img_projector is not None:
                self.img_projector = torch.compile(self.img_projector)

        self.save_hyperparameters(
            {
                "config": config.model_dump(mode="json"),
                "dataset_config": dataset_config.model_dump(mode="json"),
            },
        )

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.pop("state_dict")

        # Remove the "eeg_encoder._orig_mod.X" and replace with "eeg_encoder.X"
        for key in list(state_dict.keys()):
            if re.search(r"_orig_mod\.", key):
                new_key = re.sub(r"_orig_mod\.", "", key)
                state_dict[new_key] = state_dict.pop(key)

        checkpoint["state_dict"] = state_dict

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
            torch.save(checkpoint, temp_file.name)
            return cls.load_from_checkpoint(temp_file.name, **kwargs)

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

        projector_params = [
            {"params": self.eeg_projector.parameters(), "lr": self.config.projector_lr},
            {"params": [self.temperature], "lr": self.config.projector_lr},
        ]

        if self.img_projector is not None:
            projector_params.append(
                {
                    "params": self.img_projector.parameters(),
                    "lr": self.config.projector_lr,
                }
            )

        projector_optimizer = torch.optim.Adam(
            projector_params,
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

    @cached_property
    def num_train_batches(self) -> int:
        """Return the length of the training dataloader."""
        return len(self.data_module.train_dataloader())

    def forward(self, img_latent: torch.Tensor, eeg_data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        eeg_latent = self.eeg_encoder(eeg_data)
        eeg_latent = self.eeg_projector(eeg_latent)

        # Normalize the image latents if they are not already normalized
        if not self.config.embed_normalized:
            img_latent = nn.functional.normalize(img_latent, dim=-1)

        if self.img_projector is not None:
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

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

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

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_top1_acc", top1_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_top3_acc", top3_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_top5_acc", top5_acc, prog_bar=False, on_step=False, on_epoch=True)

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

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_top1_acc", top1_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_top3_acc", top3_acc, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log(
            "test_top5_acc", top5_acc, prog_bar=False, on_step=False, on_epoch=True
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
