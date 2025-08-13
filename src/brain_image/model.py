from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import cached_property, lru_cache
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal, cast
from matplotlib.pyplot import bar
from pydantic import BaseModel, field_validator
from sympy import O
import torch
import torch.nn as nn
import einops
import math
import itertools as it
import lightning as pl
import tqdm
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from brain_image.configs import BaseConfig, get_device, get_device_str
from brain_image.data import (
    EEGDataModule,
    EEGDatasetConfig,
    TensorCache,
    batch_load_images,
    get_image_paths,
    load_image_from_path,
    preprocess_image,
)
import dreamsim
from dreamsim.model import PerceptualModel

from brain_image.reconstruction import ReconstructionPipeline
from brain_image.utils import get_dtype
from brain_image.prior import BrainDiffusionPrior, DiffusionPriorNetwork

task_type_options = ["align", "recon"]
model_name_options = ["synclr", "clip"]
patch_size_options = ["16", "32"]
aligned_options = ["aligned", "unaligned"]
normalize_options = ["norm", "unnorm"]

default_aligned_option = "aligned"
default_patch_size = "16"
default_normalize_option = "unnorm"

recon_model_options = ["sd_highlevel", "sd_lowlevel"]


def extract_model_config(task_type: str, model_config_str: str) -> dict[str, str]:
    if task_type not in task_type_options:
        raise ValueError(f"Invalid task type: {task_type}")
    if task_type == "recon":
        assert model_config_str in recon_model_options
        return {
            "task_type": "recon",
            "model_name": model_config_str,
        }

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

    return {
        "task_type": "align",
        "aligned_option": aligned_option,
        "model_name": model_name,
        "patch_size": patch_size,
        "normalize_option": normalize_option,
    }


def load_image_encoder(
    task_type: str,
    model_config_str: str,
    models_path: Path,
    download_weights: bool = True,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float16,
    img_size: tuple[int, int] = (224, 224),
    compile: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    model_config = extract_model_config(task_type, model_config_str)
    if device is None:
        device = get_device()
    if isinstance(device, str):
        device = torch.device(device)

    if model_config["task_type"] == "recon":
        model_name = model_config["model_name"]
        pipe = ReconstructionPipeline.from_stable_diffusion(
            dtype=dtype,
            device=device,
        )
        if compile:
            pipe.compile()

        if model_name == "sd_highlevel":

            def embed(imgs: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    imgs = imgs.to(device=device)
                    latent = pipe.encode_conditioning_image(imgs)
                    latent = latent.detach().cpu()
                return latent

            return embed

        elif model_name == "sd_lowlevel":

            def embed(imgs: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    imgs = imgs.to(device=device)
                    latent = pipe.encode_low_level_image(imgs)

                    latent = latent.detach().cpu()
                return latent

            return embed

        else:
            raise ValueError(f"Invalid model name: {model_name}")

    # Align
    aligned_option = model_config["aligned_option"]
    model_name = model_config["model_name"]
    patch_size = model_config["patch_size"]
    normalize_option = model_config["normalize_option"]

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
            normalize_embeds=False,
            stride=patch_size,
            load_dir=str(models_path),
            baseline=True,
            device=device or get_device_str(),   # type: ignore
        )

    else:
        model, _ = dreamsim.dreamsim(
            dreamsim_type=model_type,
            cache_dir=str(models_path),
            normalize_embeds=False,
            device=device or get_device_str(),   # type: ignore
        )

    model = model.to(device=device, dtype=dtype)
    model.eval().requires_grad_(False)
    if compile:
        model = torch.compile(model)

    logging.info(f"Model {model_name} loaded successfully.")

    def embed(imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            imgs = imgs.to(device=device)
            imgs = preprocess_image(imgs, img_size=list(img_size)).to(
                device=device, dtype=dtype
            )

            latent = model.embed(imgs)  # type: ignore
            latent = latent.detach().cpu()
        return latent

    return embed


class EEGEncoderConfig(BaseConfig):
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


class EEGEncoder(nn.Module):
    def __init__(
        self,
        config: EEGEncoderConfig = EEGEncoderConfig(),
    ):
        # Adapted from https://github.com/eeyhsong/NICE-EEG
        super(EEGEncoder, self).__init__()

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
        hidden_dim: int = 768,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.l_proj = nn.Linear(embed_dim, hidden_dim)
        self.l_inner = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.l_out = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x = self.l_proj(x)
        x = self.l_inner(x) + x_res
        x = self.norm1(x)
        x = self.l_out(x)

        return x


class EEGAlignmentConfig(BaseConfig):
    align_target_model: str = "unaligned_synclr_16"
    low_recon_model: str = "sd_lowlevel"
    high_recon_model: str = "sd_highlevel"
    do_align: bool = True
    do_low_recon: bool = False
    do_high_recon: bool = False

    align_loss_factor: float = 1.0
    prior_loss_factor: float = 0.01
    recon_loss_factor: float = 0.01
    project_image: bool = False

    eeg_latent_dim: int = 1440
    img_latent_dim: int = 768
    project_dim: int = 768

    temperature_init: float = math.log(1 / 0.07)



class NICEConfig(EEGAlignmentConfig):
    eeg_config: EEGEncoderConfig = EEGEncoderConfig()

    encoder_lr: float = 1e-3
    projector_lr: float = 1e-3
    prior_lr: float = 1e-3

    encoder_min_lr: float = 1e-5
    projector_min_lr: float = 1e-5
    prior_min_lr: float = 1e-5

    encoder_warmup_epochs: int = 1
    projector_warmup_epochs: int = 1
    prior_warmup_epochs: int = 1

    lr_scheduler: Literal["none", "cosine_anneal"] = "cosine_anneal"
    betas: tuple[float, float] = (0.9, 0.999)

    max_epochs: int = 100

    warmup_start_frac: float = 0.1
    data_seed: int = 42

    @cached_property
    def latent_config(self) -> dict[str, str]:
        return extract_model_config("align", self.align_target_model)

    @cached_property
    def embed_normalized(self) -> bool:
        return self.latent_config.get("normalize_option") == "norm"

    @field_validator("eeg_config", mode="before")
    @classmethod
    def validate_eeg_config(cls, v):
        """Convert dict to EEGEncoderConfig if needed."""
        if isinstance(v, dict):
            return EEGEncoderConfig.model_validate(v)
        return v


class EEGAlignmentModel(ABC, pl.LightningModule):
    def __init__(
        self,
        config: EEGAlignmentConfig | dict[str, Any],
        dataset_config: EEGDatasetConfig | dict[str, Any],
        tensor_cache: TensorCache,
        dtype: torch.dtype = torch.float16,
        init_weights: bool = True,
        preload_latents: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.automatic_optimization = (
            False  # Disable automatic optimization, we will handle it manually
        )

        self.params = {
            "config": config,
            "dataset_config": dataset_config,
            "tensor_cache": tensor_cache,
            "dtype": dtype,
            "preload_latents": preload_latents,
            **kwargs,
        }

        if isinstance(config, dict):
            config = EEGAlignmentConfig.model_validate(config)

        if isinstance(dataset_config, dict):
            dataset_config = EEGDatasetConfig.model_validate(dataset_config)

        self.data_module = EEGDataModule(dataset_config)

        self.tensor_cache = tensor_cache
        self.config = config

        self.temperature = nn.Parameter(
            torch.tensor(config.temperature_init, dtype=torch.float32)
        )
        self.loss = nn.CrossEntropyLoss()

        if init_weights:
            self._init_normal_weights()

        self.prior: BrainDiffusionPrior | None = None


        if config.do_high_recon:
            # TODO: Add diffusion prior config to these configs
            prior_net = DiffusionPriorNetwork(
                dim=config.project_dim,
            )
            self.prior = BrainDiffusionPrior(
                net=prior_net,
                image_embed_dim=config.project_dim,
            )

        if preload_latents:
            self._preload_latents()

    def _preload_latents(self, parallel: bool = True):
        def preload_latent(path: Path, split: Literal["train", "val", "test"]):
            if self.config.do_align:
                self._get_image_latent_from_cache(
                    path, "align", self.config.align_target_model, split
                )
            if self.config.do_low_recon:
                self._get_image_latent_from_cache(
                    path, "recon", self.config.low_recon_model, split
                )
            if self.config.do_high_recon:
                self._get_image_latent_from_cache(
                    path, "recon", self.config.high_recon_model, split
                )

        paths = [
            (path, split)
            for split in ["train", "test"]
            for path in get_image_paths(
                self.data_module.config.data_path / self.data_module.config.imgs_dir,
                split=cast(Literal["train", "test"], split),
            )
        ]

        if parallel:
            with ThreadPoolExecutor() as executor:
                logging.info(f"Preloading latents in parallel with {executor._max_workers} workers")
                outs = executor.map(preload_latent, *zip(*paths))
                num_items = sum(1 for _ in outs)
                logging.info(f"Preloaded {num_items} latents")
        else:
            for path, split in tqdm.tqdm(paths, desc="Preloading latents"):
                preload_latent(path, cast(Literal["train", "val", "test"], split))

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

    @cached_property
    def num_val_batches(self) -> int:
        """Return the length of the validation dataloader."""
        return len(self.data_module.val_dataloader())

    @cached_property
    def num_test_batches(self) -> int:
        """Return the length of the test dataloader."""
        return len(self.data_module.test_dataloader())

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str | Path, undo_compile: bool = False, **kwargs):
        print(checkpoint_path)
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not undo_compile:
            return super().load_from_checkpoint(checkpoint_path, **kwargs)
        
        state_dict = checkpoint.pop("state_dict")

        for key in list(state_dict.keys()):
            if re.search(r"_orig_mod\.", key):
                new_key = re.sub(r"_orig_mod\.", "", key)
                state_dict[new_key] = state_dict.pop(key)

        checkpoint["state_dict"] = state_dict

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
            torch.save(checkpoint, temp_file.name)
            return cls.load_checkpoint(temp_file.name, undo_compile=False, compile=False, **kwargs)

    @abstractmethod
    def get_similarity(
        self, img_latent: torch.Tensor, eeg_data: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, img_latent: torch.Tensor, eeg_data: torch.Tensor) -> torch.Tensor:
        return self.get_similarity(img_latent, eeg_data)

    def get_align_loss(
        self, align_image_latent: torch.Tensor, eeg_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        align_sim = compute_similarity(
            eeg_latent=eeg_latent,
            img_latent=align_image_latent,
            temperature=self.temperature,
        )

        align_loss = compute_cross_entropy_loss(align_sim)
        return align_loss, align_sim

    def get_top_n_accuracy(self, sim: torch.Tensor, n: int = 1) -> float:
        """Compute top-n accuracy."""
        labels = torch.arange(sim.size(0), device=sim.device)
        # Ensure n doesn't exceed batch size
        n = min(n, sim.size(0))
        top_n = sim.topk(n, dim=-1).indices

        correct = top_n == labels.unsqueeze(1)
        return (correct.any(dim=-1).float().sum() / correct.size(0)).item()

    @abstractmethod
    def get_brain_encoder(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_brain_projector(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_img_align_projector(self) -> nn.Module:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        loss, outputs, metrics = self.run_step(batch, batch_idx, "train")

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

    def validation_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        loss, outputs, metrics = self.run_step(batch, batch_idx, "val")

        if batch_idx == self.num_val_batches - 1:
            self.evaluate_reconstructions(batch, batch_idx, "val")

        return loss, outputs, metrics
    
    @torch.no_grad()
    def evaluate_reconstructions(self, batch, batch_idx, stage: Literal["val", "test"], log_images: bool = True, num_reconstructions: int = 3):
        reconstructions = self.get_reconstructions(batch, batch_idx, stage, num_reconstructions=num_reconstructions)
        if reconstructions is None:
            return
                    
        if log_images:
            wandb_logger = self.get_wandb_logger()
            if wandb_logger is not None:
                wandb_logger.log_image(
                key=f"{stage}_recon",
                images=[recon.detach().cpu().float() for recon in reconstructions],
            )

        image_paths = [Path(path) for path in batch["img_path"][:num_reconstructions]]
        images = batch_load_images(image_paths).to(reconstructions.device, dtype=reconstructions.dtype)

        lpips_score = self._get_lpips_score(reconstructions, images)
        self.log(f"{stage}_recon_lpips", lpips_score, prog_bar=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def _get_lpips_score(self, reconstructions: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        lpips.requires_grad_(False)
        lpips.to(self.device)

        # Prepare images for lpips - Need to be in the [-1, 1] range and same shape as reconstructions
        reconstructions = reconstructions*2-1
        images = images/255.0
        images = torch.nn.functional.interpolate(images, reconstructions.shape[-2:], mode="bicubic")
        images = torch.clamp(images, min=0, max=1)
        images = images*2-1

        return lpips(reconstructions, images)

    def test_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        loss, outputs, metrics = self.run_step(batch, batch_idx, "test")

        if batch_idx == self.num_test_batches - 1:
            self.evaluate_reconstructions(batch, batch_idx, "test")

        return loss, outputs, metrics
    
    def get_wandb_logger(self) -> WandbLogger | None:
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None


    @torch.no_grad()
    def get_reconstructions(self, batch, batch_idx, stage: Literal["val", "test"], num_reconstructions: int = 3) -> torch.Tensor | None:
        if self.prior is None:
            return None

        batch_size = num_reconstructions        

        dtype = self.dtype if isinstance(self.dtype, torch.dtype) else get_dtype(self.dtype)
        device = self.device if isinstance(self.device, torch.device) else get_device(self.device)

        eeg_data = batch["eeg_data"][:batch_size].to(device, dtype=dtype)
        eeg_embed = self.get_brain_encoder()(eeg_data)
        eeg_proj = self.get_brain_projector()(eeg_embed)

        prior_pred = self.prior.p_sample_loop_ddpm(torch.Size([batch_size, self.config.img_latent_dim]), brain_embedding=eeg_proj, dtype=dtype)

        pipe = ReconstructionPipeline.from_stable_diffusion(dtype=dtype, device=device)
        imgs_reconstructed = pipe.reconstruct_latents(prior_pred)
        del pipe

        return imgs_reconstructed


    def run_step(
        self, batch, batch_idx, stage: Literal["train", "val", "test"]
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        batch = self.prepare_batch(batch, batch_idx, stage)

        dtype = (
            self.dtype if isinstance(self.dtype, torch.dtype) else get_dtype(self.dtype)
        )
        device = (
            self.device
            if isinstance(self.device, torch.device)
            else get_device(self.device)
        )

        eeg_data = batch["eeg_data"].to(device, dtype=dtype)
        eeg_latent = self.get_brain_encoder()(eeg_data)
        proj_eeg_latent = self.get_brain_projector()(eeg_latent)

        loss = torch.zeros((1,), device=device, dtype=dtype, requires_grad=True)
        metrics = {}
        outputs = {}

        on_step = stage == "train"

        with torch.no_grad() if (stage == "val" or stage == "test") else nullcontext():
            if self.config.do_align:
                align_image_latent = batch["align_image_latent"].to(device, dtype=dtype)

                if self.config.project_image:
                    align_image_latent = self.get_img_align_projector()(align_image_latent)

                align_loss, align_sim = self.get_align_loss(
                    align_image_latent, proj_eeg_latent
                )
                align_loss = align_loss * self.config.align_loss_factor
                loss = loss + align_loss

                metrics.update({"align_loss": align_loss})

                outputs.update(
                    {
                        "align_sim": align_sim,
                        "align_image_latent": align_image_latent,
                    }
                )

                self.log(
                    f"{stage}_align_loss",
                    align_loss,
                    prog_bar=False,
                    on_step=on_step,
                    on_epoch=True,
                )

                if stage == "val" or stage == "test":
                    top1_acc = self.get_top_n_accuracy(align_sim, n=1)
                    top3_acc = self.get_top_n_accuracy(align_sim, n=3)
                    top5_acc = self.get_top_n_accuracy(align_sim, n=5)

                    metrics.update(
                        {
                            "align_loss": align_loss,
                            "top1_acc": top1_acc,
                            "top3_acc": top3_acc,
                            "top5_acc": top5_acc,
                        }
                    )

                    self.log(
                        f"{stage}_top1_acc",
                        top1_acc,
                        prog_bar=True,
                        on_step=on_step,
                        on_epoch=True,
                    )
                    self.log(
                        f"{stage}_top3_acc",
                        top3_acc,
                        prog_bar=False,
                        on_step=on_step,
                        on_epoch=True,
                    )
                    self.log(
                        f"{stage}_top5_acc",
                        top5_acc,
                        prog_bar=False,
                        on_step=on_step,
                        on_epoch=True,
                    )

            if self.config.do_high_recon:
                assert self.prior is not None, "Prior is not initialized"

                high_recon_image_latent = batch["high_recon_image_latent"].to(device, dtype=dtype)

                # Note: We use the projected EEG latent here because the original latent is not same dim as images
                prior_loss, prior_pred = self.prior.forward(
                    brain_embedding=proj_eeg_latent,        
                    image_embedding=high_recon_image_latent,
                )
                prior_loss = prior_loss * self.config.prior_loss_factor
                loss = loss + prior_loss

                recon_loss = torch.nn.functional.mse_loss(high_recon_image_latent, prior_pred)
                recon_loss = recon_loss * self.config.recon_loss_factor
                loss = loss + recon_loss

                metrics.update({"prior_loss": prior_loss})

                outputs.update({"prior_pred": prior_pred, "recon_loss": recon_loss})

                self.log(
                    f"{stage}_prior_loss",
                    prior_loss,
                    prog_bar=False,
                    on_step=on_step,
                    on_epoch=True,
                )

                self.log(
                    f"{stage}_recon_loss",
                    recon_loss,
                    prog_bar=False,
                    on_step=on_step,
                    on_epoch=True,
                )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)

        return loss, outputs, metrics

    def _get_image_latent_from_cache(
        self, img_path: Path, *model_config: str
    ) -> torch.Tensor:
        return self.tensor_cache.get(str(img_path), *model_config)

    def _get_batch_from_cache(
        self, img_paths: list[Path], *model_config: str
    ) -> torch.Tensor:
        return torch.stack(
            [
                self._get_image_latent_from_cache(img_path, *model_config)
                for img_path in img_paths
            ]
        )

    def prepare_batch(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> dict[str, Any]:
        img_paths = batch["img_path"]
        eeg_data = batch["eeg_data"]
        device = eeg_data.device

        if stage == "val":
            stage = "test"

        if self.config.do_align:
            align_image_latent = self._get_batch_from_cache(
                img_paths, "align", self.config.align_target_model, stage
            )
            batch["align_image_latent"] = align_image_latent.to(
                device=device, dtype=eeg_data.dtype
            )

        if self.config.do_low_recon:
            low_recon_image_latent = self._get_batch_from_cache(
                img_paths, "recon", self.config.low_recon_model, stage
            )
            batch["low_recon_image_latent"] = low_recon_image_latent.to(
                device=device, dtype=eeg_data.dtype
            )

        if self.config.do_high_recon:
            high_recon_image_latent = self._get_batch_from_cache(
                img_paths, "recon", self.config.high_recon_model, stage
            )
            batch["high_recon_image_latent"] = high_recon_image_latent.to(
                device=device, dtype=eeg_data.dtype
            )

        return batch


class NICEModel(EEGAlignmentModel):
    def __init__(
        self,
        config: NICEConfig | dict[str, Any],
        dataset_config: EEGDatasetConfig | dict[str, Any] = EEGDatasetConfig(),
        compile: bool = True,
        modules_to_compile: list[str] = ["eeg_encoder", "eeg_projector", "align_img_projector", "prior"],
        cache_dir: Path | None = None,
        preload_latents: bool = True,
        dtype: torch.dtype = torch.float16,
        init_weights: bool = True,
    ):
        super(NICEModel, self).__init__(
            config=config,
            dataset_config=dataset_config,
            tensor_cache=TensorCache(cache_path=cache_dir or Path("cache/tensorcache")),
            dtype=dtype,
            init_weights=init_weights,
            preload_latents=preload_latents,
        )

        if isinstance(config, dict):
            config = NICEConfig.model_validate(config)

        self.config = config

        if not config.project_image and (config.project_dim != config.img_latent_dim):
            raise ValueError(
                "Projected dimension must match the image latent dimension if project_image is False"
            )

        
        self.eeg_encoder = EEGEncoder(config.eeg_config)
        self.eeg_projector = LatentProjector(
            embed_dim=config.eeg_latent_dim,
            proj_dim=config.project_dim,
        )
        
        self.align_img_projector = (
            LatentProjector(
                embed_dim=config.img_latent_dim,
                proj_dim=config.project_dim,
            )
            if config.project_image
            else None
        )

        if compile:
            logging.info("Compiling model...")
            for module in modules_to_compile:
                match module:
                    case "eeg_encoder":
                        self.eeg_encoder = torch.compile(self.eeg_encoder)
                    case "eeg_projector":
                        self.eeg_projector = torch.compile(self.eeg_projector)
                    case "align_img_projector":
                        self.align_img_projector = torch.compile(self.align_img_projector)
                    case "prior":
                        self.prior = cast(BrainDiffusionPrior, torch.compile(self.prior))
                    case _:
                        raise ValueError(f"Unknown module to compile: {module}")

        self.save_hyperparameters(
            {
                "config": config.model_dump(mode="json"),
                "dataset_config": self.data_module.config.model_dump(mode="json"),
            },
        )

        self.compile: bool = compile
        self.modules_to_compile: list[str] = modules_to_compile

    def get_brain_encoder(self) -> nn.Module:
        return cast(nn.Module, self.eeg_encoder)

    def get_brain_projector(self) -> nn.Module:
        return cast(nn.Module, self.eeg_projector)

    def get_img_align_projector(self) -> nn.Module:
        return cast(nn.Module, self.align_img_projector)

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        # TODO: Refactor

        encoder_optimizer = torch.optim.Adam(
            self.eeg_encoder.parameters(),
            lr=self.config.encoder_lr,
            betas=self.config.betas,
        )

        projector_params = [
            {"params": self.eeg_projector.parameters(), "lr": self.config.projector_lr},
            {"params": [self.temperature], "lr": self.config.projector_lr},
        ]

        if self.align_img_projector is not None:
            projector_params.append(
                {
                    "params": self.align_img_projector.parameters(),
                    "lr": self.config.projector_lr,
                }
            )

        projector_optimizer = torch.optim.Adam(
            projector_params,
            betas=self.config.betas,
        )

        prior_optimizer = torch.optim.Adam(
            self.prior.parameters(),
            lr=self.config.prior_lr,
            betas=self.config.betas,
        ) if self.prior is not None else None

        encoder_schedulers = []
        projector_schedulers = []
        prior_schedulers = []

        projector_milestones = []
        encoder_milestones = []
        prior_milestones = []

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

        if self.config.prior_warmup_epochs > 0 and self.prior is not None:
            assert prior_optimizer is not None, "Prior optimizer is not initialized"
            prior_schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    prior_optimizer,
                    start_factor=self.config.warmup_start_frac,
                    total_iters=self.config.prior_warmup_epochs,
                )
            )
            prior_milestones.append(self.config.prior_warmup_epochs)

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
                if prior_optimizer is not None:
                    prior_schedulers.append(
                        torch.optim.lr_scheduler.ConstantLR(
                            prior_optimizer,
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
                if prior_optimizer is not None:
                    prior_schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            prior_optimizer,
                            T_max=self.config.max_epochs,
                            eta_min=self.config.prior_min_lr,
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

        if prior_optimizer is not None:
            prior_scheduler = (
                torch.optim.lr_scheduler.SequentialLR(
                    prior_optimizer,
                    schedulers=prior_schedulers,
                    milestones=prior_milestones,
                )
            )
        else:
            prior_scheduler = None

        optimizer_configs = [
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
            }
        ]
        if prior_optimizer is not None:
            optimizer_configs.append(
                {
                    "optimizer": prior_optimizer,
                    "lr_scheduler": prior_scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            )

        return optimizer_configs

    def get_similarity(
        self, img_latent: torch.Tensor, eeg_data: torch.Tensor
    ) -> torch.Tensor:
        eeg_latent = self.get_brain_encoder()(eeg_data)
        eeg_latent = self.get_brain_projector()(eeg_latent)

        img_latent = self.get_img_align_projector()(img_latent)

        sim = compute_similarity(
            eeg_latent=eeg_latent,
            img_latent=img_latent,
            temperature=self.temperature,
        )

        return sim


@torch.compile
def compute_cross_entropy_loss(sim: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss."""
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_e = nn.functional.cross_entropy(sim, labels)
    loss_i = nn.functional.cross_entropy(sim.T, labels)
    loss = (loss_e + loss_i) / 2
    return loss


@torch.compile
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
