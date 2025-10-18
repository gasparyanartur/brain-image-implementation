from collections.abc import Iterator
import math
import lightning as pl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from lightning.pytorch.loggers import WandbLogger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import itertools as it
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import PIL
from PIL import Image
from functools import cached_property
from pathlib import Path
from brain_image.configs import BaseConfig, get_device
from brain_image.data import (
    EEGDataModule,
    EEGDatasetConfig,
    EmbeddingsMap,
    TensorCache,
    batch_load_images,
    get_image_paths,
)
from brain_image.model.eeg_encoder import create_eeg_encoder
from brain_image.model.eeg_encoder.eeg_encoder import EEGEncoder
from brain_image.model.loss import CLIPLoss, InfoNCELoss

from brain_image.model.prior import (
    DiffusionPriorConfig,
    SimpleDiffusionPrior,
)


from typing import Any, Literal, Mapping, Sequence, TypedDict, cast
import logging
from brain_image.reconstruction import (
    IPAdapterReconstructionPipeline,
    ReconstructionPipeline,
)
from brain_image.utils import (
    DTYPE,
    current_fig_to_img,
    find_module_content_in_state_dict,
    gather_dataloader,
    get_dtype,
    get_mean_gradients,
    get_norm_dir_len,
    key_in_dict,
)

import tqdm
import re
import tempfile
import time
import wandb


class EEGAlignmentConfig(BaseConfig):
    align_img_encoder: str = "unaligned_synclr_vitb16"
    recon_latent_encoder: str = "sd_variations_v2"
    prior_img_encoder: str = "clip_vitl14"
    eeg_encoder: str = "nice"

    do_align: bool = True
    do_recon_low: bool = False
    do_recon: bool = True

    align_input_noise: float = 0.0

    plot_lowdim_proj: bool = True
    low_dim_proj_pca: int = 50

    debug_prior_use_target_as_cond: bool = False

    align_loss_type: Literal["clip", "infonce"] = "infonce"
    align_loss_epoch: int = 0
    align_loss_factor: float = 0.1
    align_mse_loss_factor: float = 10.0
    prior_loss_factor: float = 1.0

    full_eval_every_epochs: int = 1
    skip_eval_first_epoch: bool = True

    img_latent_dim: int = 768
    project_dim: int = 768

    num_reconstructions: int = 3
    temperature_init: float = 0.07
    log_gradients: bool = False

    prior: DiffusionPriorConfig | None = DiffusionPriorConfig()

    encoder_lr: float = 3e-4
    prior_lr: float = 3e-4

    encoder_min_lr: float = 1e-5
    prior_min_lr: float = 1e-6

    encoder_warmup_epochs: int = 1
    prior_warmup_epochs: int = 5

    encoder_delay_epochs: int = 0
    prior_delay_epochs: int = 0

    lr_scheduler: Literal["none", "cosine_anneal"] = "cosine_anneal"
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01

    debug_metrics: bool = False
    eeg_encoder_path: Path | None = None

    max_epochs: int = 100

    seed: int = 42

    prog_bar_metrics: list[str] = [
        "train/loss",
        "val/loss",
        "val/align/mse_loss",
        "val/align/clip_loss",
        "val/prior/loss",
    ]

    embeddings_to_compute_stats: list[str] = ["prior_img_latent"]
    modules_to_compile: list[str] = [
        "eeg_encoder",
        "prior",
    ]

    modules_to_train: list[str] = [
        "eeg_encoder",
        "prior",
    ]


class DataBatchT(TypedDict):
    img_path: list[str] | None
    eeg_data: torch.Tensor | None
    sub: torch.Tensor | None
    idx: torch.Tensor | None
    eeg_latent: torch.Tensor | None
    eeg_latent_normed: torch.Tensor | None
    align_img_latent: torch.Tensor | None
    prior_img_latent: torch.Tensor | None
    recon_latent: torch.Tensor | None
    prior_pred: torch.Tensor | None
    prior_pred_single: torch.Tensor | None


class EEGAlignmentModel(pl.LightningModule):
    def __init__(
        self,
        config: EEGAlignmentConfig,
        dataset_config: EEGDatasetConfig,
        dtype: torch.dtype = DTYPE,
        init_weights: bool = False,
        compile: bool = True,
        cache_dir: Path = Path("tensorcache"),
        eeg_encoder_path: Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.automatic_optimization = (
            False  # Disable automatic optimization, we will handle it manually
        )

        self.config: EEGAlignmentConfig = (
            config
            if isinstance(config, EEGAlignmentConfig)
            else EEGAlignmentConfig.model_validate(config)
        )

        tensor_cache = TensorCache(cache_dir)
        embeddings_map: EmbeddingsMap = {
            "align_img_latent": (
                self.config.align_img_encoder if self.config.do_align else None
            ),
            "prior_img_latent": (
                self.config.prior_img_encoder if self.config.do_recon else None
            ),
            "recon_latent": (
                self.config.recon_latent_encoder if self.config.do_recon_low else None
            ),
        }

        self.data_module = EEGDataModule(
            dataset_config, tensor_cache=tensor_cache, embeddings_map=embeddings_map
        )

        self.temperature = nn.Parameter(
            torch.tensor(self.config.temperature_init, dtype=torch.float32)
        )
        self.loss = nn.CrossEntropyLoss()

        if init_weights:
            self._init_normal_weights()

        self.prior: SimpleDiffusionPrior | None = None
        if self.config.do_recon:
            assert self.config.prior, "Prior config must be provided"
            self.prior = SimpleDiffusionPrior(self.config.prior).to(dtype)

        elif self.config.do_recon_low:
            raise ValueError(
                "Cannot do low level reconstruction in without high level reconstruction"
            )

        eeg_encoder_path = eeg_encoder_path or self.config.eeg_encoder_path
        self.eeg_encoder = create_eeg_encoder(
            self.config.eeg_encoder,
            checkpoint_path=eeg_encoder_path,
        )

        if self.config.do_align:
            if self.config.align_loss_type == "clip":
                self.align_loss = CLIPLoss(self.config.temperature_init)
            elif self.config.align_loss_type == "infonce":
                self.align_loss = InfoNCELoss(self.config.temperature_init)
            else:
                raise ValueError(
                    f"Unknown align_loss_type: {self.config.align_loss_type}"
                )
        else:
            self.align_loss = None

        if compile:
            compile_kwargs = {}

            logging.info(f"Compiling model with kwargs: {compile_kwargs}")

            for module in self.config.modules_to_compile:
                match module:
                    case "eeg_encoder":
                        if self.eeg_encoder is not None:
                            self.eeg_encoder.compile(**compile_kwargs)
                    case "prior":
                        if self.prior is not None:
                            self.prior.compile(**compile_kwargs)
                    case "align_loss":
                        if self.align_loss is not None:
                            self.align_loss.compile(**compile_kwargs)
                    case _:
                        raise ValueError(f"Unknown module to compile: {module}")

        self.save_hyperparameters(
            {
                "config": self.config.model_dump(mode="json"),
                "dataset_config": self.data_module.config.model_dump(mode="json"),
            },
        )

        self.compile: bool = compile

        self.learning_rate_options: list[dict[str, Any]] = []

        self.atleast_one_training_step: bool = False
        self.embedding_stats = self._get_embeddings_stats()

    def configure_optimizers(self):
        def _iter_params(*models: nn.Module | None):
            param_list: list[nn.Parameter] = []
            for model in models:
                if model is not None:
                    param_list.extend(model.parameters())
            return iter(param_list)

        @dataclass
        class OptimizerConfig:
            name: str
            modules: list[nn.Module | None]
            lr: float
            min_lr: float
            warmup_epochs: int
            delay_epochs: int
            enabled: bool

        optimizer_configs = [
            OptimizerConfig(
                name="eeg_encoder",
                modules=[self.eeg_encoder, self.align_loss],
                lr=self.config.encoder_lr,
                min_lr=self.config.encoder_min_lr,
                warmup_epochs=self.config.encoder_warmup_epochs,
                delay_epochs=self.config.encoder_delay_epochs,
                enabled=self.eeg_encoder is not None,
            ),
            OptimizerConfig(
                name="prior",
                modules=[self.prior],
                lr=self.config.prior_lr,
                min_lr=self.config.prior_min_lr,
                warmup_epochs=self.config.prior_warmup_epochs,
                delay_epochs=self.config.prior_delay_epochs,
                enabled=self.prior is not None,
            ),
        ]

        optimizer_options = []
        num_train_batches = self.data_module.get_num_batches("train")
        for config in optimizer_configs:
            modules_to_opt = cast(
                list[nn.Module], [m for m in config.modules if m is not None]
            )

            if (
                (config.name not in self.config.modules_to_train)
                or (not config.enabled)
                or (not modules_to_opt)
            ):
                config.enabled = False
                for module in modules_to_opt:
                    logging.info("Disabling optimization of module: %s")
                    module.requires_grad_(False)
                continue

            logging.info(
                f"Creating optimizer: {config.name} - lr: {config.lr}, min_lr: {config.min_lr}, warmup_epochs: {config.warmup_epochs}, delay_epochs: {config.delay_epochs}"
            )
            warmup_steps = config.warmup_epochs * num_train_batches
            delay_steps = config.delay_epochs * num_train_batches
            total_steps = self.config.max_epochs * num_train_batches

            optimizer = torch.optim.AdamW(
                _iter_params(*modules_to_opt),
                lr=config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
            schedulers = []
            milestones = []

            if delay_steps > 0:
                schedulers.append(
                    torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 0)
                )
                milestones.append(delay_steps + max(milestones or [0]))

            if warmup_steps > 0:
                schedulers.append(
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        total_iters=warmup_steps,
                    )
                )
                milestones.append(warmup_steps + max(milestones or [0]))

            if self.config.lr_scheduler == "cosine_anneal":
                schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=total_steps - max(milestones or [0]),
                        eta_min=config.min_lr,
                    )
                )
            elif self.config.lr_scheduler == "none":
                schedulers.append(
                    torch.optim.lr_scheduler.ConstantLR(
                        optimizer,
                        factor=1.0,
                    )
                )
            else:
                raise ValueError(f"Unknown lr_scheduler: {self.config.lr_scheduler}")

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=milestones,
            )

            optimizer_options.append(
                {
                    "name": config.name,
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            )

        self.learning_rate_options = optimizer_options
        return optimizer_options

    def _init_normal_weights(self):
        # These are the weight configurations used in MindsEye. Probably optional.
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

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: str | Path, undo_compile: bool = False, **kwargs
    ):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not undo_compile:
            return super().load_from_checkpoint(checkpoint_path, **kwargs)

        # When compile with torch.compile(model) instead of model.compile(),
        # the state_dict gets nested under orig_mod. Needs to be removed.

        state_dict = checkpoint.pop("state_dict")

        for key in list(state_dict.keys()):
            if re.search(r"_orig_mod\.", key):
                new_key = re.sub(r"_orig_mod\.", "", key)
                state_dict[new_key] = state_dict.pop(key)

        checkpoint["state_dict"] = state_dict

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
            torch.save(checkpoint, temp_file.name)
            return cls.load_checkpoint(
                temp_file.name, undo_compile=False, compile=False, **kwargs
            )

    def training_step(self, batch, batch_idx):
        self.atleast_one_training_step = True

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for opt in optimizers:
            opt.zero_grad()

        losses, metrics, outputs = self.run_step(batch, batch_idx, "train")

        self.manual_backward(losses["loss"])

        for opt in optimizers:
            opt.step()

        for scheduler in schedulers:
            if scheduler is None:
                continue

            scheduler.step()  # type: ignore

        metrics["lr/step"] = scheduler.last_epoch if scheduler is not None else -1
        for opt_option in self.learning_rate_options:
            metrics["lr/" + opt_option["name"]] = opt_option[
                "lr_scheduler"
            ].get_last_lr()[0]

        if self.config.log_gradients:
            with torch.no_grad():
                for name, module in [
                    ("eeg_encoder", self.eeg_encoder),
                    ("prior", self.prior),
                ]:
                    if module is None:
                        continue

                    grads = get_mean_gradients(module)
                    if grads is not None:
                        metrics["grad/" + name] = grads

        for metric_name, metric_value in it.chain(losses.items(), metrics.items()):
            name = (
                f"train/{metric_name}" if "/" not in metric_name[:10] else metric_name
            )
            self.log(
                name,
                metric_value,
                prog_bar=name in self.config.prog_bar_metrics,
                on_step=True,
                on_epoch=False,
            )

        return losses["loss"]

    def validation_step(
        self, batch, batch_idx
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        losses, outputs, metrics = self.run_step(batch, batch_idx, "val")

        if self.atleast_one_training_step and (
            batch_idx == self.data_module.get_num_batches("val") - 1
        ):
            eval_metrics, eval_outputs = self.run_full_validation(split="val")
            metrics.update(eval_metrics)

            if eval_outputs and ((wandb_logger := self.get_wandb_logger()) is not None):
                for k, v in eval_outputs.items():
                    prefix = "val/" if k.split("/")[0] != "debug" else ""
                    wandb_logger.log_image(key=f"{prefix}{k}", images=v)

        for metric_name, metric_value in it.chain(losses.items(), metrics.items()):
            prefix = "val/" if metric_name.split("/")[0] != "debug" else ""
            name = f"{prefix}{metric_name}"
            self.log(
                name,
                metric_value,
                prog_bar=name in self.config.prog_bar_metrics,
                on_step=False,
                on_epoch=True,
            )

        return losses, outputs, metrics

    def test_step(
        self, batch, batch_idx
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        losses, outputs, metrics = self.run_step(batch, batch_idx, "test")

        if self.atleast_one_training_step and (
            batch_idx == self.data_module.get_num_batches("test") - 1
        ):
            eval_metrics, eval_outputs = self.run_full_validation(split="test")
            metrics.update(eval_metrics)

            if eval_outputs and ((wandb_logger := self.get_wandb_logger()) is not None):
                for k, v in eval_outputs.items():
                    wandb_logger.log_image(key="test/" + k, images=v)

        for metric_name, metric_value in it.chain(losses.items(), metrics.items()):
            self.log(
                f"test/{metric_name}",
                metric_value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        return losses, outputs, metrics

    def run_step(
        self,
        batch,
        batch_idx,
        stage: Literal["train", "val", "test"],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        device = (
            self.device
            if isinstance(self.device, torch.device)
            else get_device(self.device)
        )

        eeg_latent, eeg_latent_normed = self.get_eeg_latent(
            batch,
            device,
            self.dtype,
        )

        losses: dict[str, torch.Tensor] = {}
        metrics = {}
        outputs = {}
        cache = {
            "eeg_latent": eeg_latent,
            "eeg_latent_normed": eeg_latent_normed,
        }

        with torch.no_grad() if (stage == "val" or stage == "test") else nullcontext():
            if self.config.do_align:
                self._run_step_do_align(
                    batch=batch,
                    losses=losses,
                    metrics=metrics,
                    outputs=outputs,
                    cache=cache,
                    device=device,
                    stage=stage,
                )

            if self.config.do_recon and (
                (self.current_epoch >= self.config.prior_delay_epochs)
                or (
                    stage != "train"
                )  # On training step, no need to run this if it's not time
            ):
                self._run_step_do_recon(
                    batch=batch,
                    losses=losses,
                    metrics=metrics,
                    outputs=outputs,
                    cache=cache,
                    stage=stage,
                    device=device,
                )

        loss = torch.stack(list(losses.values())).sum()
        losses["loss"] = loss

        return losses, outputs, metrics

    def _run_step_do_align(
        self,
        batch: Mapping[str, Any],
        losses: dict[str, Any],
        metrics: dict[str, Any],
        outputs: dict[str, Any],
        cache: dict[str, Any],
        stage,
        device,
    ):
        assert (
            eeg_latent_normed := cache.get("eeg_latent_normed")
        ) is not None, "EEG latent is not initialized"
        assert (
            align_img_latent := batch.get("align_img_latent")
        ) is not None, "Align image latent is not in batch"
        assert (idx := batch.get("idx")) is not None, "Index is not initialized"
        assert self.align_loss is not None, "Align loss is not initialized"

        with torch.no_grad():
            align_img_latent_normed = F.normalize(align_img_latent.to(device))

        if stage == "train" and self.config.align_input_noise > 0:
            align_img_latent_normed = align_img_latent_normed + (
                torch.randn_like(align_img_latent_normed)
                * align_img_latent_normed.std(dim=-1, keepdim=True)
                * self.config.align_input_noise
            )

        with torch.no_grad():
            # There might be duplicates (different subjects, same image)
            # labels = (idx.unsqueeze(0) == idx.unsqueeze(1)).float()
            labels = None

        align_clip_loss, align_logits = self.align_loss(
            eeg_latent_normed, align_img_latent_normed, labels=labels
        )

        align_clip_loss = (
            align_clip_loss
            * self.config.align_loss_factor
            * (self.current_epoch >= self.config.align_loss_epoch)
        )
        align_mse_loss = (
            torch.nn.functional.mse_loss(eeg_latent_normed, align_img_latent_normed)
            * self.config.align_mse_loss_factor
        )

        losses.update(
            {
                "align/mse_loss": align_mse_loss,
                "align/clip_loss": align_clip_loss,
            }
        )

        if stage == "val":
            align_logits = align_logits.detach()

            with torch.no_grad():
                metrics.update(
                    {
                        "align/top1": (
                            align_logits.topk(1, dim=-1).indices.flatten()
                            == idx.detach()
                        )
                        .float()
                        .mean()
                        .cpu(),
                        "align/cos": align_logits.diag().mean().cpu(),
                        "align/logit_scale": self.align_loss.logit_scale.detach().cpu(),
                    }
                )

    def _run_step_do_recon(
        self,
        batch: Mapping[str, Any],
        losses: dict[str, Any],
        metrics: dict[str, Any],
        outputs: dict[str, Any],
        cache: dict[str, Any],
        stage,
        device,
    ):
        assert self.config.prior is not None, "Prior config is not initialized"
        assert self.prior is not None, "Prior is not initialized"
        assert (
            target_latent := batch["prior_img_latent"]
        ) is not None, "Prior image latent is not defined"
        assert (
            eeg_latent_normed := cache["eeg_latent"]
        ) is not None, "Normed eeg latent is not defined"

        # target_latent = F.normalize(target_latent)
        with torch.no_grad():
            match self.config.prior.norm_scheme:
                case "none":
                    pass
                case "z_scale":
                    stats = self.embedding_stats["prior_img_latent"]
                    target_latent = (
                        target_latent - stats["mean"].to(target_latent.device)
                    ) / stats["std"].to(target_latent.device)
                case "l2_scale":
                    target_latent = F.normalize(target_latent) * (
                        target_latent.dim(-1) ** 0.5
                    )


        if self.config.debug_prior_use_target_as_cond:
            eeg_latent_normed = F.normalize(-target_latent)
            # eeg_latent_normed = F.normalize(_index_encoding(batch["idx"]))

        batch_size = target_latent.size(0)

        noise = torch.randn_like(target_latent)
        timesteps = torch.randint(
            0,
            self.config.prior.num_training_timesteps,
            size=(batch_size,),
            device=device,
        )
        noisy_latent = self.prior.scheduler.add_noise(
            target_latent, noise, timesteps=cast(torch.IntTensor, timesteps)
        )

        noise_pred = self.prior.forward(
            noisy_latent,
            timesteps,
            eeg_latent_normed,
            self.config.prior.cond_drop_prob,
        )
        prior_loss = (
            torch.nn.functional.mse_loss(noise_pred, noise)
            * self.config.prior_loss_factor
        )
        pred = self.prior.remove_noise(noisy_latent, noise_pred, timesteps)
        sim_loss = (
            1
            - torch.linalg.vecdot(
                F.normalize(target_latent), F.normalize(pred), dim=-1
            ).mean()
        )
        pred_mse = torch.nn.functional.mse_loss(pred, target_latent) * 0.01
        losses.update(
            {
                "prior/sim_loss": sim_loss,
                "prior/noise_mse_loss": prior_loss,
                "prior/pred_mse_loss": pred_mse,
            }
        )

        if stage == "val":
            timesteps_50 = int(
                self.prior.config.num_training_timesteps * 0.5
            ) * torch.ones(batch_size, device=device, dtype=torch.int32)
            noisy_latent_50 = self.prior.scheduler.add_noise(
                target_latent, noise, timesteps=cast(torch.IntTensor, timesteps_50)
            )
            noise_pred_50 = self.prior.forward(
                noisy_latent_50,
                timesteps_50,
                eeg_latent_normed,
            )
            pred_50 = self.prior.predict_step(
                noise_pred_50, timesteps_50, eeg_latent_normed
            )
            metrics.update(
                {
                    "prior/pred/cos": torch.linalg.vecdot(
                        F.normalize(target_latent), F.normalize(pred_50), dim=-1
                    )
                    .mean()
                    .cpu(),
                }
            )
            if self.config.debug_metrics:
                metrics.update(
                    {
                        "debug/prior/pred/mean": pred_50.mean(dim=-1).mean().cpu(),
                        "debug/prior/pred/std": pred_50.std(dim=-1).mean().cpu(),
                        "debug/prior/pred/norm": pred_50.norm(dim=-1).mean().cpu(),
                    }
                )

        losses.update(
            {
                "prior/loss": prior_loss,
            }
        )

        if stage == "val":
            if self.config.debug_metrics:
                metrics.update(
                    {
                        "debug/prior/target/mean": target_latent.mean(dim=-1)
                        .mean()
                        .cpu(),
                        "debug/prior/target/std": target_latent.std(dim=-1)
                        .mean()
                        .cpu(),
                        "debug/prior/target/norm": target_latent.norm(dim=-1)
                        .mean()
                        .cpu(),
                        "debug/prior/noise/mean": noise.mean(dim=-1).mean().cpu(),
                        "debug/prior/noise/std": noise.std(dim=-1).mean().cpu(),
                        "debug/prior/noise/norm": noise.norm(dim=-1).mean().cpu(),
                        "debug/prior/noise_pred/mean": noise_pred.mean(dim=-1)
                        .mean()
                        .cpu(),
                        "debug/prior/noise_pred/std": noise_pred.std(dim=-1)
                        .mean()
                        .cpu(),
                        "debug/prior/noise_pred/norm": noise_pred.norm(dim=-1)
                        .mean()
                        .cpu(),
                    }
                )

    @torch.no_grad()
    def run_full_validation(
        self, split: Literal["val", "test"]
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        is_not_first_or_flag_set = (self.current_epoch > 0) or (
            not self.config.skip_eval_first_epoch
        )
        is_right_mod = self.current_epoch % self.config.full_eval_every_epochs == 0
        is_right_val_epoch = is_not_first_or_flag_set and is_right_mod

        if not (is_right_val_epoch or split == "test"):
            return {}, {}

        metrics = {}
        img_outputs = {}

        data_loader = self.data_module.get_dataloader(split)
        all_data = gather_dataloader(data_loader)
        all_data = cast(DataBatchT, all_data)

        eeg_latent, eeg_latent_normed = self.get_all_eeg_latents(
            all_data,
            batch_size=self.data_module.config.get_batch_size(split),
            progress_bar=False,
            normalize=True,
        )

        all_data["eeg_latent"], all_data["eeg_latent_normed"] = (
            eeg_latent,
            eeg_latent_normed,
        )
        device, dtype = self._get_device_dtype()

        if self.config.do_align:
            metrics_align, img_outputs_align = self._run_validation_align(
                all_data, device
            )
            metrics.update(metrics_align)
            img_outputs.update(img_outputs_align)

        if self.config.do_recon:
            metrics_prior, img_outputs_prior = self._run_validation_recon(
                all_data, device, split
            )
            metrics.update(metrics_prior)
            img_outputs.update(img_outputs_prior)

        if self.config.plot_lowdim_proj:
            metrics_plot, img_outputs_plot = self.plot_lowdim_projection(all_data)
            metrics.update(metrics_plot)
            img_outputs.update(img_outputs_plot)

        return metrics, img_outputs

    @torch.no_grad()
    def _run_validation_align(
        self,
        all_data: DataBatchT,
        device: torch.device,
    ):
        assert (
            eeg_latent_normed := all_data.get("eeg_latent_normed")
        ) is not None, "EEG latent is not in batch"

        assert (
            img_paths := all_data.get("img_path")
        ) is not None, "Image paths are not in batch"

        assert (
            align_img_latent := all_data.get("align_img_latent")
        ) is not None, "Align img latent is not in batch"

        assert (indexes := all_data.get("idx")) is not None, "Indices are not in batch"

        eeg_latent_normed = eeg_latent_normed.to(device)
        align_img_latent = align_img_latent.to(device)
        indexes = indexes.to(device)

        align_img_latent_normed = F.normalize(align_img_latent)
        sim = eeg_latent_normed.to(device) @ align_img_latent_normed.T
        top_sim = sim.topk(1, dim=-1).indices.flatten()  # <B, 1>
        chosen_idx = indexes[top_sim]
        top1_acc = (chosen_idx == indexes).float().mean()

        target_img_paths = [Path(img_paths[i]) for i in indexes[:3]]
        chosen_img_paths = [Path(img_paths[i]) for i in chosen_idx[:3]]
        chosen_imgs = batch_load_images(chosen_img_paths)

        eeg_align_cos = torch.linalg.vecdot(
            eeg_latent_normed, align_img_latent_normed, dim=-1
        )

        metrics = {
            "eval/align/top1_acc": top1_acc,
            "eval/align/eeg_cos": eeg_align_cos.detach().mean().cpu(),
        }

        img_outputs = {
            "eval/align/chosen": [x.detach().cpu().float() for x in chosen_imgs],
            "eval/align/target": target_img_paths,
        }
        return metrics, img_outputs

    def _run_validation_recon(
        self,
        all_data: DataBatchT,
        device: torch.device,
        split: Literal["val", "test"] = "val",
    ):
        assert (target := all_data.get("prior_img_latent")) is not None
        stats = self.embedding_stats["prior_img_latent"]
        target = (target - stats["mean"].to(target)) / stats["std"].to(target)

        metrics = {}
        img_outputs = {}

        gen = torch.Generator(device).manual_seed(self.config.seed)
        pred = self.get_all_prior_preds(
            all_data,
            progress_bar=True,
            generator=gen,
            batch_size=self.data_module.config.get_batch_size(split),
        )
        assert pred is not None

        all_data["prior_pred"] = pred

        if self.current_epoch == 0 and self.config.skip_eval_first_epoch:
            pass
        else:
            metrics_recon, img_outputs_recon = self._evaluate_reconstructions(
                all_data,
                split,
                num_reconstructions=self.config.num_reconstructions,
            )
            metrics.update(metrics_recon)
            img_outputs.update(img_outputs_recon)

        metrics.update(
            {
                "eval/prior/pred_mean": pred.mean(dim=0).mean().cpu(),
                "eval/prior/pred_std": pred.std(dim=0).mean().cpu(),
                "eval/prior/recon_cos": torch.linalg.vecdot(
                    F.normalize(pred), F.normalize(target), dim=-1
                )
                .mean()
                .cpu(),
                "eval/prior/target_to_pred_ratio": (
                    target.norm(p=2, dim=-1).mean() / pred.norm(p=2, dim=-1).mean()
                ).cpu(),
            }
        )

        return metrics, img_outputs

    @torch.no_grad()
    def plot_lowdim_projection(
        self, all_data: DataBatchT
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        pca = PCA(n_components=self.config.low_dim_proj_pca)
        tsne = TSNE(n_components=2)

        latents_highdim = []
        labels = []
        c = []

        if self.config.do_align:
            assert (
                align_img_latent := all_data.get("align_img_latent")
            ) is not None, "Align image latent is not in batch"
            assert (
                eeg_latent_normed := all_data.get("eeg_latent_normed")
            ) is not None, "EEG latent is not in batch"

            align_img_latent_normed = F.normalize(align_img_latent)
            n = align_img_latent_normed.size(0)

            latents_highdim.extend([eeg_latent_normed, align_img_latent_normed])
            labels.extend(["eeg_latent", "align_target_latent"])
            c.extend(["blue", "red"])

        if self.config.do_recon:
            assert (
                prior_img_latent := all_data.get("prior_img_latent")
            ) is not None, "Prior image latent is not in batch"
            assert (
                prior_pred := all_data.get("prior_pred")
            ) is not None, "Prior pred is not in batch"
            n = prior_img_latent.size(0)

            latents_highdim.extend(
                [F.normalize(prior_img_latent), F.normalize(prior_pred)]
            )
            labels.extend(["prior_img_latent", "prior_pred"])
            c.extend(["green", "orange"])

        if not latents_highdim:
            return {}, {}

        latents_highdim = torch.cat(latents_highdim, dim=0).detach().cpu().numpy()

        logging.info(f"Doing PCA to {self.config.low_dim_proj_pca} dimensions")
        t1 = time.time()
        latents_middim = pca.fit_transform(latents_highdim)
        logging.info(f"Doing TSNE to 2 dimensions")
        latents_lowdim = tsne.fit_transform(latents_middim)
        t2 = time.time()

        for i in range(len(labels)):
            plt.scatter(
                latents_lowdim[i * n : (i + 1) * n, 0],
                latents_lowdim[i * n : (i + 1) * n, 1],
                c=c[i],
                label=labels[i],
            )
        plt.legend()

        plot_image = current_fig_to_img()
        logging.info(f"Finished projecting latents in {t2 - t1:.3f} seconds")

        metrics = {}
        img_outputs = {
            "plot/lowdim": [plot_image],
        }

        return metrics, img_outputs

    @torch.no_grad()
    def _evaluate_reconstructions(
        self,
        batch,
        stage: Literal["val", "test"],
        num_reconstructions: int = 5,
    ):
        recon_pred, recon_target = self._get_reconstructions(
            batch, stage, num_reconstructions=num_reconstructions
        )

        if recon_pred is None or recon_target is None:
            return {}, {}

        lpips_score = self._get_lpips_score(recon_pred, recon_target)

        metrics = {
            f"eval/recon/lpips": lpips_score.detach().cpu(),
        }
        img_outputs = {
            "eval/recon/pred": [x.detach().cpu().float() for x in recon_pred],
            "eval/recon/target": [x.detach().cpu().float() for x in recon_target],
        }

        return metrics, img_outputs

    @torch.no_grad()
    def _get_lpips_score(
        self, img_recon: torch.Tensor, img_target: torch.Tensor
    ) -> torch.Tensor:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")
        lpips.requires_grad_(False)
        lpips.to(self.device)

        # Prepare images for lpips - Need to be in the [-1, 1] range and same shape as reconstructions
        img_recon = F.normalize(img_recon - img_recon.min()) * 2 - 1
        img_target = F.normalize(img_target - img_recon.min()) * 2 - 1

        return lpips(img_recon, img_target)

    def get_wandb_logger(self) -> WandbLogger | None:
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    @torch.no_grad()
    def _get_reconstructions(
        self,
        batch,
        stage: Literal["val", "test"],
        num_reconstructions: int = 5,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.prior is None:
            return None, None

        device, dtype = self._get_device_dtype()

        target_latent = batch["prior_img_latent"][:num_reconstructions].to(
            device, dtype=dtype
        )
        target_latent = F.normalize(target_latent)
        prior_pred = batch["prior_pred"][:num_reconstructions].to(device, dtype=dtype)
        conditioning = torch.cat(
            [F.normalize(prior_pred), F.normalize(target_latent)], dim=0
        )

        pipe = IPAdapterReconstructionPipeline.load_pretrained(device=device)
        reconstruction = pipe.reconstruct_latents(conditioning)
        del pipe

        recon_imgs, recon_target = torch.chunk(reconstruction, 2, dim=0)
        return recon_imgs, recon_target

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def get_eeg_latent(
        self,
        batch: DataBatchT,
        device,
        dtype,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not normalize:
            raise ValueError(f"Non-normalized EEG not supported")

        assert self.eeg_encoder is not None, "EEG encoder is not initialized"
        assert (subs := batch.get("sub")) is not None, "Subject is not in batch"
        assert (
            eeg_data := batch.get("eeg_data")
        ) is not None, "EEG data is not in batch"

        eeg_data = eeg_data.to(device, dtype=dtype)
        subs = subs.to(device)
        eeg_latent = self.eeg_encoder(eeg_data, subs)
        eeg_latent_normed = F.normalize(eeg_latent)

        return eeg_latent, eeg_latent_normed

    def _get_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        dtype = (
            self.dtype if isinstance(self.dtype, torch.dtype) else get_dtype(self.dtype)
        )
        device = (
            self.device
            if isinstance(self.device, torch.device)
            else get_device(self.device)
        )
        return device, dtype

    @torch.no_grad()
    def get_all_prior_preds(
        self,
        all_data: DataBatchT,
        batch_size: int = 512,
        progress_bar: bool = True,
        generator: torch.Generator | None = None,
        prior_kwargs: dict = {},
    ) -> torch.Tensor | None:
        if self.prior is None:
            return None

        assert (
            eeg_latent_normed := all_data.get("eeg_latent_normed")
        ) is not None, "EEG latent is not in batch"

        if self.config.debug_prior_use_target_as_cond:
            assert (
                target_latent := all_data.get("prior_img_latent")
            ) is not None, "Prior image latent is not in batch"
            assert (
                "idx" in all_data and all_data["idx"] is not None
            ), "Index is not in batch"
            eeg_latent_normed = F.normalize(-target_latent)

        device, dtype = self._get_device_dtype()

        n = eeg_latent_normed.size(0)
        all_prior_preds_ = []
        with tqdm.tqdm(
            total=n,
            desc="Prior sampling",
            disable=not progress_bar,
        ) as pbar:
            for i in range(0, n, batch_size):
                eeg_values = eeg_latent_normed[i : i + batch_size].to(device)

                prior_pred = self.prior.generate(
                    conditioning=eeg_values,
                    generator=generator,
                    **prior_kwargs,
                )
                
                match self.prior.config.norm_scheme:
                    case "none":
                        pass
                    case "z_scale":
                        stats = self.embedding_stats["prior_img_latent"]
                        prior_pred = prior_pred * stats["std"].to(prior_pred.device) + stats["mean"].to(prior_pred.device)
                    case "l2_scale":
                        prior_pred = F.normalize(prior_pred, dim=-1) * (prior_pred.size(-1) ** 0.5)

                all_prior_preds_.append(prior_pred.detach().cpu())
                pbar.update(prior_pred.size(0))

        prior_preds = torch.cat(all_prior_preds_, dim=0)
        return prior_preds

    @torch.no_grad()
    def get_all_eeg_latents(
        self,
        all_data: DataBatchT,
        batch_size: int = 512,
        progress_bar: bool = True,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            "idx" in all_data and all_data["idx"] is not None
        ), "Index is not in batch"

        device, dtype = self._get_device_dtype()
        all_latents = []
        all_latents_normed = []
        num_samples = len(all_data["idx"])

        with tqdm.tqdm(
            total=num_samples, desc="EEG encoding", disable=not progress_bar
        ) as pbar:
            for i in range(0, num_samples, batch_size):
                batch_data = cast(
                    DataBatchT,
                    {
                        k: v[i : i + batch_size]
                        for k, v in all_data.items()
                        if isinstance(v, (torch.Tensor, list))
                    },
                )
                eeg_latent, eeg_latent_normed = self.get_eeg_latent(
                    batch_data,
                    device,
                    dtype,
                    normalize=normalize,
                )
                all_latents.append(eeg_latent.detach().cpu())
                all_latents_normed.append(eeg_latent_normed.detach().cpu())
                pbar.update(len(eeg_latent))

        return torch.cat(all_latents, dim=0), torch.cat(all_latents_normed, dim=0)

    @torch.no_grad()
    def _get_embeddings_stats(self):
        logging.info(
            f"Getting embedding stats for {self.config.embeddings_to_compute_stats}"
        )
        _running_embeddings = {}

        for batch in self.train_dataloader():
            for emb_name in self.config.embeddings_to_compute_stats:
                if emb_name not in batch:
                    continue

                if emb_name not in _running_embeddings:
                    _running_embeddings[emb_name] = []

                _running_embeddings[emb_name].append(batch[emb_name])

        _running_latents = {
            k: torch.cat(v, dim=0) for k, v in _running_embeddings.items()
        }

        if "prior_img_latent" in _running_latents:
            _running_latents["prior_img_latent_normed"] = F.normalize(
                _running_latents["prior_img_latent"]
            )

        embedding_stats: dict[str, dict[str, torch.Tensor]] = {
            k: {"mean": v.mean(dim=0), "std": v.std(dim=0)}
            for k, v in _running_latents.items()
        }

        logging.info(f"Finished getting embedding stats")
        return embedding_stats


def _index_encoding(x: torch.Tensor, d_embed: int = 1024) -> torch.Tensor:
    batch_size = x.size(0)
    latent = torch.zeros((batch_size, d_embed), device=x.device, dtype=torch.float32)

    rng = torch.Generator()
    for i in range(batch_size):
        rng.manual_seed(int(x[i]))
        latent[i] = torch.randn(d_embed, generator=rng)

    return latent
