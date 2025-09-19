import lightning as pl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import itertools as it
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from brain_image.configs import BaseConfig, get_device
from brain_image.data import EEGDataModule, EEGDatasetConfig, TensorCache, batch_load_images, get_image_paths
from brain_image.model.eeg_encoder import EEGEncoder, EEGEncoderConfig
from brain_image.model.loss import CLIPLoss, InfoNCELoss
from brain_image.model.model import LatentProjector, ResidualAdapter, normalize_projection
from brain_image.model.prior import BrainDiffusionPrior, BrainDiffusionPriorConfig


from typing import Any, Literal, TypedDict, cast
import logging
from brain_image.reconstruction import ReconstructionPipeline
from brain_image.utils import DTYPE, gather_dataloader, get_dtype, get_mean_gradients, get_norm_dir_len, key_in_dict

import tqdm
import re
import tempfile
import time


class EEGAlignmentConfig(BaseConfig):
    align_target_model: str = "unaligned_synclr_16"
    low_recon_model: str = "sd_lowlevel"
    high_recon_model: str = "sd_highlevel"
    do_align: bool = True
    do_low_recon: bool = False
    do_high_recon: bool = True

    align_input_noise: float = 0.002

    use_embed_adapter: bool = False
    use_prior_adapter: bool = False
    plot_lowdim_proj: bool = True
    low_dim_proj_pca: int = 50

    align_loss_type: Literal["clip", "infonce"] = "infonce"
    align_loss_epoch: int = 0
    align_loss_factor: float = 1.
    align_mse_loss_factor: float = 0.5
    align_cos_loss_factor: float = 0.05
    prior_loss_factor: float = 0.01
    prior_sim_loss_factor: float = 1.0
    prior_len_loss_factor: float = 0.5

    project_image: bool = False
    rescale_proj_by_mean: bool = False
    norm_eeg_latent: bool = True

    full_eval_every_epochs: int = 1
    skip_eval_first_epoch: bool = True

    img_latent_dim: int = 768
    project_dim: int = 768

    prior_debug_mode: bool = (
        False  # If True, will use the target image in the prior and disable alignment
    )
    num_reconstructions: int = 5

    temperature_init: float = 0.04
    log_gradients: bool = False

    eeg_config: EEGEncoderConfig = EEGEncoderConfig()
    prior_config: BrainDiffusionPriorConfig | None = BrainDiffusionPriorConfig()

    encoder_lr: float = 1e-3
    projector_lr: float = 1e-3
    prior_lr: float = 3e-4
    embed_adapter_lr: float = 1e-3
    prior_adapter_lr: float = 1e-3
    align_loss_lr: float = 3e-4

    encoder_min_lr: float = 1e-5
    projector_min_lr: float = 1e-5
    prior_min_lr: float = 1e-6
    embed_adapter_min_lr: float = 1e-5
    prior_adapter_min_lr: float = 1e-5
    align_loss_min_lr: float = 1e-7

    encoder_warmup_epochs: int = 1
    projector_warmup_epochs: int = 1
    prior_warmup_epochs: int = 1
    embed_adapter_warmup_epochs: int = 1
    prior_adapter_warmup_epochs: int = 1
    align_loss_warmup_epochs: int = 1

    encoder_delay_epochs: int = 0
    projector_delay_epochs: int = 0
    prior_delay_epochs: int = 0
    embed_adapter_delay_epochs: int = 0
    prior_adapter_delay_epochs: int = 0
    align_loss_delay_epochs: int = 0

    lr_scheduler: Literal["none", "cosine_anneal"] = "cosine_anneal"
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01

    #freeze_eeg_encoder: bool = False

    max_epochs: int = 100

    warmup_start_frac: float = 0.35
    data_seed: int = 42

    prog_bar_metrics: list[str] = [
        "TRAIN__loss",
        "VAL__loss",
        "VAL__align_top1_acc",
        "VAL__prior_pred_cos",
    ]


class DataBatchT(TypedDict):
    img_path: list[str] | None
    eeg_data: torch.Tensor | None
    idx: torch.Tensor | None
    eeg_latent: torch.Tensor | None
    align_image_latent: torch.Tensor | None
    high_recon_image_latent: torch.Tensor | None
    low_recon_image_latent: torch.Tensor | None
    prior_pred: torch.Tensor | None
    prior_pred_single: torch.Tensor | None


def get_belong_group(img_path: list[str], to_float: bool = False) -> torch.Tensor:
    unique_ids = {v: k for k, v in enumerate(img_path)}

    belonings = [[] for _ in range(len(img_path))]
    for i, p in enumerate(img_path):
        belonings[unique_ids[p]].append(i)

    path_groups = torch.zeros((len(img_path), len(img_path)), dtype=torch.bool)
    for i, p in enumerate(img_path):
        uid = unique_ids[p]
        for op in belonings[uid]:
            path_groups[i, op] = True

    if to_float:
        path_groups = path_groups.float()

    return path_groups


class EEGAlignmentModel(pl.LightningModule):
    def __init__(
        self,
        config: EEGAlignmentConfig | dict[str, Any],
        dataset_config: EEGDatasetConfig | dict[str, Any],
        dtype: torch.dtype = DTYPE,
        init_weights: bool = True,
        preload_latents: bool = True,
        compile: bool = True,
        modules_to_compile: list[str] = [
            "eeg_encoder",
            "eeg_projector",
            "align_img_projector",
            "prior",
            "embed_adapter",
            "prior_adapter",
        ],
        cache_dir: Path = Path("cache/tensorcache"),
        eeg_encoder: EEGEncoder | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config: EEGAlignmentConfig = (
            config
            if isinstance(config, EEGAlignmentConfig)
            else EEGAlignmentConfig.model_validate(config)
        )

        self.data_module = (
            EEGDataModule(dataset_config)
            if isinstance(dataset_config, EEGDatasetConfig)
            else EEGDataModule(EEGDatasetConfig.model_validate(dataset_config))
        )

        self.automatic_optimization = (
            False  # Disable automatic optimization, we will handle it manually
        )

        self.tensor_cache = TensorCache(cache_dir)

        self.params = {
            "config": config,
            "dataset_config": dataset_config,
            "tensor_cache": self.tensor_cache,
            "dtype": dtype,
            "preload_latents": preload_latents,
            **kwargs,
        }

        self.temperature = nn.Parameter(
            torch.tensor(self.config.temperature_init, dtype=torch.float32)
        )
        self.loss = nn.CrossEntropyLoss()

        if self.config.prior_debug_mode:
            self.config.do_align = False
            self.config.do_low_recon = False
            self.config.do_high_recon = True

        if init_weights:
            self._init_normal_weights()

        self.prior: BrainDiffusionPrior | None = None
        self.embed_adapter: ResidualAdapter | None = (
            ResidualAdapter(latent_dim=self.config.img_latent_dim)
            if self.config.do_high_recon and self.config.use_embed_adapter
            else None
        )

        self.prior_adapter: ResidualAdapter | None = (
            ResidualAdapter(latent_dim=self.config.img_latent_dim)
            if self.config.do_high_recon and self.config.use_prior_adapter
            else None
        )

        if self.config.do_high_recon:
            assert self.config.prior_config, "Prior config must be provided"
            self.prior = BrainDiffusionPrior(self.config.prior_config).to(dtype)

        elif self.config.do_low_recon:
            raise ValueError(
                "Cannot do low level reconstruction in without high level reconstruction"
            )

        if preload_latents:
            self._preload_latents()

        if not self.config.project_image and (
            self.config.project_dim != self.config.img_latent_dim
        ):
            raise ValueError(
                "Projected dimension must match the image latent dimension if project_image is False"
            )

        self.eeg_encoder: EEGEncoder | None = (
            (eeg_encoder or EEGEncoder(self.config.eeg_config))
            if not self.config.prior_debug_mode
            else None
        )
        self.eeg_projector: LatentProjector | None = None
        self.align_img_projector: LatentProjector | None = (
            LatentProjector(
                embed_dim=self.config.img_latent_dim,
                proj_dim=self.config.project_dim,
            )
            if self.config.project_image and not self.config.prior_debug_mode
            else None
        )

        if self.config.do_align:
            if self.config.align_loss_type == "clip":
                self.align_loss = CLIPLoss(self.config.temperature_init)
            elif self.config.align_loss_type == "infonce":
                self.align_loss = InfoNCELoss(self.config.temperature_init)
            else:
                raise ValueError(f"Unknown align_loss_type: {self.config.align_loss_type}")
        else:
            self.align_loss = None

        if compile:
            compile_kwargs = {}

            logging.info(f"Compiling model with kwargs: {compile_kwargs}")

            for module in modules_to_compile:
                match module:
                    case "eeg_encoder":
                        if self.eeg_encoder is not None:
                            self.eeg_encoder.compile(**compile_kwargs)
                    case "eeg_projector":
                        if self.eeg_projector is not None:
                            self.eeg_projector.compile(**compile_kwargs)
                    case "align_img_projector":
                        if self.align_img_projector is not None:
                            self.align_img_projector.compile(**compile_kwargs)
                    case "prior":
                        if self.prior is not None:
                            self.prior.compile(**compile_kwargs)
                    case "prior_adapter":
                        if self.prior_adapter is not None:
                            self.prior_adapter.compile(**compile_kwargs)
                    case "embed_adapter":
                        if self.embed_adapter is not None:
                            self.embed_adapter.compile(**compile_kwargs)
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
        self.modules_to_compile: list[str] = modules_to_compile

        self.learning_rate_options: list[dict[str, Any]] = []

        self.atleast_one_training_step: bool = False

    def configure_optimizers(self):
        @dataclass
        class OptimizerConfig:
            name: str
            model: nn.Module | None
            lr: float
            min_lr: float
            warmup_epochs: int
            delay_epochs: int

        optimizer_configs = [
            OptimizerConfig(
                name="eeg_encoder",
                model=self.eeg_encoder,
                lr=self.config.encoder_lr,
                min_lr=self.config.encoder_min_lr,
                warmup_epochs=self.config.encoder_warmup_epochs,
                delay_epochs=self.config.encoder_delay_epochs,
            ),
            OptimizerConfig(
                name="eeg_projector",
                model=self.eeg_projector,
                lr=self.config.projector_lr,
                min_lr=self.config.projector_min_lr,
                warmup_epochs=self.config.projector_warmup_epochs,
                delay_epochs=self.config.projector_delay_epochs,
            ),
            OptimizerConfig(
                name="align_img_projector",
                model=self.align_img_projector,
                lr=self.config.projector_lr,
                min_lr=self.config.projector_min_lr,
                warmup_epochs=self.config.projector_warmup_epochs,
                delay_epochs=self.config.projector_delay_epochs,
            ),
            OptimizerConfig(
                name="prior",
                model=self.prior,
                lr=self.config.prior_lr,
                min_lr=self.config.prior_min_lr,
                warmup_epochs=self.config.prior_warmup_epochs,
                delay_epochs=self.config.prior_delay_epochs,
            ),
            OptimizerConfig(
                name="embed_adapter",
                model=self.embed_adapter,
                lr=self.config.embed_adapter_lr,
                min_lr=self.config.embed_adapter_min_lr,
                warmup_epochs=self.config.embed_adapter_warmup_epochs,
                delay_epochs=self.config.embed_adapter_delay_epochs,
            ),
            OptimizerConfig(
                name="prior_adapter",
                model=self.prior_adapter,
                lr=self.config.prior_adapter_lr,
                min_lr=self.config.prior_adapter_min_lr,
                warmup_epochs=self.config.prior_adapter_warmup_epochs,
                delay_epochs=self.config.prior_adapter_delay_epochs,
            ),
            OptimizerConfig(
                name="align_loss",
                model=self.align_loss,
                lr=self.config.align_loss_lr,
                min_lr=self.config.align_loss_min_lr,
                warmup_epochs=self.config.align_loss_warmup_epochs,
                delay_epochs=self.config.align_loss_delay_epochs,
            )
        ]
        optimizer_configs = [x for x in optimizer_configs if x.model is not None]

        optimizer_options = []
        for optimizer_config in optimizer_configs:
            warmup_steps = optimizer_config.warmup_epochs * self.num_train_batches
            delay_steps = optimizer_config.delay_epochs * self.num_train_batches
            total_steps = self.config.max_epochs * self.num_train_batches

            optimizer = torch.optim.AdamW(
                (
                    optimizer_config.model.parameters()
                    if optimizer_config.model is not None
                    else []
                ),
                lr=optimizer_config.lr,
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
                        start_factor=self.config.warmup_start_frac,
                        total_iters=warmup_steps,
                    )
                )
                milestones.append(warmup_steps + max(milestones or [0]))

            if self.config.lr_scheduler == "cosine_anneal":
                schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=total_steps - max(milestones or [0]),
                        eta_min=optimizer_config.min_lr,
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
                    "name": optimizer_config.name,
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            )

        self.learning_rate_options = optimizer_options
        return optimizer_options

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
                logging.info(
                    f"Preloading latents in parallel with {executor._max_workers} workers"
                )
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

    def train_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        return self.data_module.train_dataloader(**kwargs)

    def val_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        return self.data_module.val_dataloader(**kwargs)

    def test_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        return self.data_module.test_dataloader(**kwargs)

    @cached_property
    def num_train_batches(self) -> int:
        return len(self.data_module.train_dataloader())

    @cached_property
    def num_val_batches(self) -> int:
        return len(self.data_module.val_dataloader())

    @cached_property
    def num_test_batches(self) -> int:
        return len(self.data_module.test_dataloader())

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: str | Path, undo_compile: bool = False, **kwargs
    ):
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
            return cls.load_checkpoint(
                temp_file.name, undo_compile=False, compile=False, **kwargs
            )

    def forward(self, img_latent: torch.Tensor, eeg_data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_top_n_accuracy(
        self,
        sim: torch.Tensor,
        ns: list[int] = [1, 3, 5],
        labels: torch.Tensor | None = None,
    ) -> list[float]:
        B = sim.size(0)
        labels = (
            torch.arange(sim.size(0), device=sim.device)
            if labels is None
            else labels.to(sim.device)
        )
        ns = sorted(ns)
        max_n = min(max(ns), B)  # Largest top-n cannot exceed batch size
        top_n = sim.topk(max_n, dim=-1).indices  # <B, n>

        accuracies = []

        for n in ns:
            if n > max_n:
                break

            label_in_top_n = top_n[:, :n] == labels.unsqueeze(1)
            num_labels_found = label_in_top_n.any(dim=-1).sum()
            accuracy = num_labels_found / B
            accuracies.append(accuracy.item())

        return accuracies

    @abstractmethod
    def get_brain_encoder(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_brain_projector(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_img_align_projector(self) -> nn.Module:
        raise NotImplementedError

    def set_mode(self, mode: Literal["train", "val", "test"]):
        if mode == "train":
            if self.embed_adapter is not None:
                self.embed_adapter.train()
            if self.prior_adapter is not None:
                self.prior_adapter.train()
            if self.prior is not None:
                self.prior.train()
            if self.eeg_encoder is not None:
                self.eeg_encoder.train()
            if self.eeg_projector is not None:
                self.eeg_projector.train()
            if self.align_img_projector is not None:
                self.align_img_projector.train()

        elif mode == "val" or mode == "test":
            if self.embed_adapter is not None:
                self.embed_adapter.eval()
            if self.prior_adapter is not None:
                self.prior_adapter.eval()
            if self.prior is not None:
                self.prior.eval()
            if self.eeg_encoder is not None:
                self.eeg_encoder.eval()
            if self.eeg_projector is not None:
                self.eeg_projector.eval()
            if self.align_img_projector is not None:
                self.align_img_projector.eval()

    def training_step(self, batch, batch_idx):
        self.set_mode("train")

        self.atleast_one_training_step = True

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for opt in optimizers:
            opt.zero_grad()

        loss, outputs, metrics = self._run_step(batch, batch_idx, "train")

        self.manual_backward(loss)

        for opt in optimizers:
            opt.step()

        for scheduler in schedulers:
            if scheduler is None:
                continue

            scheduler.step()  # type: ignore

        scheduler_step = scheduler.last_epoch if scheduler is not None else -1
        self.log(
            "LR__STEP", scheduler_step, prog_bar=False, on_step=True, on_epoch=False
        )

        for opt_option in self.learning_rate_options:
            name = opt_option["name"]
            lr = opt_option["lr_scheduler"].get_last_lr()[0]
            self.log(f"LR__{name}", lr, prog_bar=False, on_step=True, on_epoch=False)

        if self.config.log_gradients:
            with torch.no_grad():
                grad_modules_to_log = [
                    ("eeg_encoder", self.eeg_encoder),
                    ("eeg_projector", self.eeg_projector),
                    ("align_img_projector", self.align_img_projector),
                    ("prior", self.prior),
                    ("embed_adapter", self.embed_adapter),
                    ("prior_adapter", self.prior_adapter)
                ]
                grad_modules_to_log = [
                    x for x in grad_modules_to_log if x[1] is not None
                ]
                for name, module in grad_modules_to_log:
                    grads = get_mean_gradients(module)
                    if grads is not None:
                        self.log(f"GRAD__{name}", grads, prog_bar=False, on_step=True, on_epoch=False)

        return loss

    def validation_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        self.set_mode("val")

        loss, outputs, metrics = self._run_step(batch, batch_idx, "val")

        if self.atleast_one_training_step and (batch_idx == self.num_val_batches - 1):
            self._run_full_validation(split="val")

            logging.info(
                f"Epoch: {self.epoch-1} -> {self.epoch}, Training step: {self.global_step}"
            )

        return loss, outputs, metrics

    @property
    def epoch(self) -> int:
        return self.current_epoch

    def test_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        self.set_mode("test")

        loss, outputs, metrics = self._run_step(batch, batch_idx, "test")

        if batch_idx == self.num_test_batches - 1:
            self._run_full_validation(split="test")

        return loss, outputs, metrics

    def _get_proj_eeg_latent(
        self,
        batch: DataBatchT,
        device,
        dtype,
        eps: float = 1e-8,
        use_align_if_debug: bool = True,
        normalize: bool = True
    ) -> torch.Tensor:
        assert "align_image_latent" in batch and batch["align_image_latent"] is not None, "Align image latent is not in batch"

        if use_align_if_debug and self.config.prior_debug_mode:
            proj_eeg_latent = batch["align_image_latent"].to(device, dtype=dtype)

        else:
            assert self.eeg_encoder is not None, "EEG encoder is not initialized"
            assert "eeg_data" in batch and batch["eeg_data"] is not None, "EEG data is not in batch"

            eeg_data = batch["eeg_data"].to(device, dtype=dtype)
            proj_eeg_latent = self.eeg_encoder(eeg_data)

            if self.eeg_projector is not None:
                proj_eeg_latent = self.eeg_projector(proj_eeg_latent)

        if normalize:
            proj_eeg_latent = normalize_projection(
                proj_eeg_latent, self.config.rescale_proj_by_mean, eps
            )

        if self.embed_adapter is not None:
            proj_eeg_latent = self.embed_adapter(proj_eeg_latent)

        return proj_eeg_latent

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
        eps: float = 1e-8,
        batch_size: int = 512,
        progress_bar: bool = True,
        generator: torch.Generator | None = None,
        prior_kwargs: dict = {},
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if self.prior is None:
            return None, None

        assert "align_image_latent" in all_data and all_data["align_image_latent"] is not None, "Align image latent is not in batch"
        assert "high_recon_image_latent" in all_data and all_data["high_recon_image_latent"] is not None, "High recon image latent is not in batch"

        device, dtype = self._get_device_dtype()
        if not key_in_dict("eeg_latent", all_data):
            all_data["eeg_latent"] = self.get_all_eeg_latents(
                all_data, batch_size=batch_size, progress_bar=progress_bar, eps=eps
            )

        assert "eeg_latent" in all_data and all_data["eeg_latent"] is not None, "EEG latent is already in batch"

        proj_eeg_latent = all_data["eeg_latent"].to(device)
        target_latent = all_data["high_recon_image_latent"].to(device)

        all_prior_preds_ = []
        all_prior_preds_single_ = []
        with tqdm.tqdm(
            total=proj_eeg_latent.size(0),
            desc="Prior sampling",
            disable=not progress_bar,
        ) as pbar:
            for i in range(0, proj_eeg_latent.size(0), batch_size):
                latent_batch = proj_eeg_latent[i : i + batch_size]
                target_batch = target_latent[i : i + batch_size]
                prior_pred = self.prior.p_sample_loop(
                    torch.Size([latent_batch.size(0), self.config.img_latent_dim]),
                    brain_embedding=latent_batch,
                    dtype=dtype,
                    progress_bar=False,
                    generator=generator,
                    cond_scale=1.0,
                    **prior_kwargs,
                )
                _, prior_prep_single = self.prior(
                    brain_embedding=latent_batch,
                    image_embedding=target_batch,
                )
                prior_pred_single = prior_prep_single / self.prior.image_embed_scale
                if self.prior_adapter:
                    prior_pred = self.prior_adapter(prior_pred)
                    prior_pred_single = self.prior_adapter(prior_prep_single)

                all_prior_preds_.append(prior_pred.detach().cpu())
                all_prior_preds_single_.append(prior_pred_single.detach().cpu())

                pbar.update(latent_batch.size(0))

        return torch.cat(all_prior_preds_, dim=0), torch.cat(all_prior_preds_single_, dim=0)

    @torch.no_grad()
    def get_all_eeg_latents(
        self,
        all_data: DataBatchT,
        batch_size: int = 512,
        progress_bar: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor | None:
        assert "idx" in all_data and all_data["idx"] is not None, "Index is not in batch"

        device, dtype = self._get_device_dtype()
        all_latents = []
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
                    }
                )
                proj_eeg_latent = self._get_proj_eeg_latent(
                    batch_data, device, dtype, eps=eps, use_align_if_debug=True, normalize=True
                )
                all_latents.append(proj_eeg_latent.detach().cpu())
                pbar.update(len(proj_eeg_latent))

        return torch.cat(all_latents, dim=0)

    @torch.no_grad()
    def plot_lowdim_projection(
        self, all_data: DataBatchT, show_plot: bool = False
    ) -> dict[str, Any]:
        assert "align_image_latent" in all_data and all_data["align_image_latent"] is not None, "Align image latent is not in batch"
        assert "eeg_latent" in all_data and all_data["eeg_latent"] is not None, "EEG latent is not in batch"
        assert "high_recon_image_latent" in all_data and all_data["high_recon_image_latent"] is not None, "High recon image latent is not in batch"

        align_latents = all_data["align_image_latent"]
        eeg_latent = all_data["eeg_latent"]
        high_recon_latent = all_data["high_recon_image_latent"]
        if self.config.do_high_recon:
            assert "prior_pred" in all_data and all_data["prior_pred"] is not None, "Prior pred is not in all data"
            prior_pred = all_data["prior_pred"]

        logging.info(
            f"Projecting latents from dim {all_data['high_recon_image_latent'].size(1)} to 2 dimensions"
        )

        n = len(high_recon_latent)

        pca = PCA(n_components=self.config.low_dim_proj_pca)
        tsne = TSNE(n_components=2)

        if self.config.do_high_recon:
            latents_highdim = torch.cat(
                [align_latents, eeg_latent, high_recon_latent, prior_pred], dim=0
            ).numpy()
            labels = [
                "align_target", "eeg_latent", "prior_target", "prior_pred"
            ]
            c = ["blue", "red", "green", "orange"]

        else:
            latents_highdim = torch.cat([align_latents, eeg_latent, high_recon_latent], dim=0).numpy()
            labels = [
                "align_target", "eeg_latent", "prior_target"
            ]
            c = ["blue", "red", "green"]


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

        fig = plt.gcf()
        if show_plot:
            fig.show()

        logging.info(f"Finished projecting latents in {t2 - t1:.3f} seconds")

        return {"VAL__lowdim_proj": fig}

    @torch.no_grad()
    def _run_full_validation(self, split: Literal["val", "test"]) -> None:
        stage_prefix = f"{split.upper()}__"

        is_not_first_or_flag_set = (self.epoch > 0) or (
            not self.config.skip_eval_first_epoch
        )
        is_right_mod = self.epoch % self.config.full_eval_every_epochs == 0
        is_right_val_epoch = is_not_first_or_flag_set and is_right_mod

        if not (is_right_val_epoch or split == "test"):
            return

        metrics = {}
        img_outputs = {}

        data_generator = torch.Generator().manual_seed(42)
        loader_kwargs = {
            "generator": data_generator,
            "persistent_workers": False,
            "pin_memory": False,
        }
        data_loader = (
            self.val_dataloader(**loader_kwargs)
            if split == "val"
            else self.test_dataloader(**loader_kwargs)
        )

        all_data = gather_dataloader(
            data_loader, lambda b: self.prepare_batch(b, stage=split)
        )
        all_data = cast(DataBatchT, all_data)

        align_image_latent = cast(torch.Tensor, all_data["align_image_latent"])
        eeg_latents = self.get_all_eeg_latents(all_data, batch_size=self.data_module.config.val_batch_size, progress_bar=False)

        all_data["eeg_latent"] = eeg_latents
        img_paths = all_data["img_path"]

        assert isinstance(
                align_image_latent, torch.Tensor
            ), "Align image latent is not a tensor"
        assert eeg_latents is not None, "EEG latents are not initialized"
        assert img_paths is not None, "Image paths are not initialized"

        device, dtype = self._get_device_dtype()

        if self.config.do_align:
            if self.config.project_image:
                assert (
                    self.align_img_projector is not None
                ), "Image projector is not initialized"
                align_image_latent = self.align_img_projector(align_image_latent)

            align_image_latent_normed = normalize_projection(
                align_image_latent, self.config.rescale_proj_by_mean
            )

            sim = eeg_latents @ align_image_latent_normed.T
            top_sim = sim.topk(1, dim=-1).indices.flatten()  # <B, 1>

            chosen_img_paths = [img_paths[i] for i in top_sim]
            num_correct = sum(
                [x == y for x, y in zip(chosen_img_paths, img_paths)]
            )
            top1_acc = num_correct / len(img_paths)

            eeg_align_cos = torch.linalg.vecdot(eeg_latents, align_image_latent_normed, dim=-1)

            img_outputs.update(
                {
                    f"align_test_target": [wandb.Image(x) for x in img_paths[:3]],  # type: ignore
                    f"align_test_chosen": [wandb.Image(x) for x in chosen_img_paths[:3]],  # type: ignore
                }
            )
            metrics.update(
                {
                    "eval_align_top1_acc": top1_acc,
                    "eval_align_eeg_cos": eeg_align_cos.mean().cpu(),
                }
            )

        if self.config.do_high_recon:
            assert "high_recon_image_latent" in all_data and all_data["high_recon_image_latent"] is not None, "High recon image latent is not in batch"

            gen = torch.Generator(device).manual_seed(42)
            all_prior_preds, all_prior_preds_single = self.get_all_prior_preds(
                all_data,
                progress_bar=True,
                generator=gen,
                batch_size=self.data_module.config.val_batch_size,
            )
            assert all_prior_preds is not None
            assert all_prior_preds_single is not None

            # Now we compare average align latents and EEG
            all_data["prior_pred"] = all_prior_preds
            all_data["prior_pred_single"] = all_prior_preds_single

            if self.epoch == 0 and self.config.skip_eval_first_epoch:
                pass
            else:
                self.evaluate_reconstructions(
                    all_data,
                    split,
                    num_reconstructions=self.config.num_reconstructions,
                )

            if self.config.plot_lowdim_proj:
                metrics.update(self.plot_lowdim_projection(all_data))

            # Compute diagnostic metrics
            prior_pred_info = get_norm_dir_len(all_data["prior_pred"])
            prior_pred_single_info = get_norm_dir_len(all_data["prior_pred_single"])
            high_recon_latents_info = get_norm_dir_len(all_data["high_recon_image_latent"])

            prior_high_recon_cos = torch.linalg.vecdot(
                prior_pred_info.dir, high_recon_latents_info.dir, dim=-1
            ).mean()
            prior_single_high_recon_cos = torch.linalg.vecdot(
                prior_pred_single_info.dir, high_recon_latents_info.dir, dim=-1
            ).mean()

            metrics.update(
                {
                    "eval_prior_high_recon_cos": prior_high_recon_cos.cpu(),
                    "eval_prior_single_high_recon_cos": prior_single_high_recon_cos.cpu(),
                    "eval_prior_pred_len": prior_pred_info.len.cpu(),
                    "eval_prior_pred_single_len": prior_pred_single_info.len.cpu(),
                    "eval_high_recon_latent_len": high_recon_latents_info.len.cpu(),
                }
            )

        if (wandb_logger := self.get_wandb_logger()) is not None:
            if metrics:
                wandb_logger.log_metrics(
                    {stage_prefix + k: v for k, v in metrics.items()}
                )
            if img_outputs:
                for k, v in img_outputs.items():
                    wandb_logger.log_image(key=stage_prefix + k, images=v)

    def _run_step(
        self,
        batch,
        batch_idx,
        stage: Literal["train", "val", "test"],
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        batch = self.prepare_batch(batch, stage)

        # TODO: Domain classifier (reduce modality gap)?
        # TODO: Try 1 subject (use best subject)

        stage_prefix = f"{stage.upper()}__"

        device = (
            self.device
            if isinstance(self.device, torch.device)
            else get_device(self.device)
        )

        proj_eeg_latent = self._get_proj_eeg_latent(
            batch, device, self.dtype, eps=eps, use_align_if_debug=False, normalize=False
        )
        proj_eeg_latent_normed = normalize_projection(
            proj_eeg_latent, self.config.rescale_proj_by_mean, eps
        )

        losses: dict[str, torch.Tensor] = {}
        metrics = {}
        outputs = {}

        on_step = stage == "train"

        with torch.no_grad() if (stage == "val" or stage == "test") else nullcontext():
            if self.config.do_align:
                assert proj_eeg_latent is not None, "EEG latent is not initialized"
                assert proj_eeg_latent_normed is not None, "EEG latent is not initialized"
                assert self.align_loss is not None, "Align loss is not initialized"
                assert "align_image_latent" in batch and batch["align_image_latent"] is not None
                assert "img_path" in batch and batch["img_path"] is not None

                align_image_latent = batch["align_image_latent"].to(device)

                if self.config.project_image:
                    assert (
                        self.align_img_projector is not None
                    ), "Image projector is not initialized"
                    align_image_latent = self.align_img_projector(align_image_latent)

                align_image_latent_normed = normalize_projection(
                    align_image_latent, self.config.rescale_proj_by_mean, eps
                )

                if stage == "train":
                    align_image_latent_normed = align_image_latent_normed + (
                        torch.rand_like(align_image_latent_normed)
                        * self.config.align_input_noise
                    )

                with torch.no_grad():
                    # There might be duplicates (different subjects, same image)
                    # So we will look at the img_path to make sure
                    labels = get_belong_group(batch["img_path"], to_float=False).to(
                        align_image_latent.device
                    )

                align_clip_loss, align_sim = self.align_loss(
                    proj_eeg_latent_normed, align_image_latent_normed, labels=labels
                )

                align_clip_loss = align_clip_loss * self.config.align_loss_factor * (self.epoch >= self.config.align_loss_epoch)
                align_mse_loss = (
                    torch.nn.functional.mse_loss(align_image_latent, proj_eeg_latent)
                    * self.config.align_mse_loss_factor
                )
                align_cos = align_sim.diag()
                align_cos_loss = (1-align_cos).mean() * self.config.align_cos_loss_factor


                losses.update(
                    {
                        "align_mse_loss": align_mse_loss,
                        "align_clip_loss": align_clip_loss,
                        "align_cos_loss": align_cos_loss
                    }
                )

                with torch.no_grad():
                    metrics.update(
                        {
                            "align_cos": align_cos.mean().detach().cpu(),
                            "align_loss_logit_scale": self.align_loss.logit_scale.detach().cpu(),
                        }
                    )

                    outputs.update(
                        {
                            "align_sim": align_sim.detach().cpu(),
                            "align_image_latent": align_image_latent.detach().cpu(),
                        }
                    )

            if self.config.do_high_recon and (
                (self.epoch >= self.config.prior_delay_epochs)
                or (
                    stage != "train"
                )  # On training step, no need to run this if it's not time
            ):
                assert self.prior is not None, "Prior is not initialized"
                assert "high_recon_image_latent" in batch and batch["high_recon_image_latent"] is not None, "High recon image latent is not in batch"

                target_latent = cast(
                    torch.Tensor, batch["high_recon_image_latent"].to(device)
                )

                target_latent_norm = target_latent.norm(dim=-1, keepdim=True)
                target_latent_dir = target_latent / (target_latent_norm + eps)

                # Note: We use the projected EEG latent here because the original latent is not same dim as images
                prior_loss, prior_pred = self.prior(
                    brain_embedding=proj_eeg_latent_normed,
                    image_embedding=target_latent,
                )
                prior_pred = prior_pred / self.prior.image_embed_scale
                prior_loss = prior_loss * self.config.prior_loss_factor

                if self.prior_adapter is not None:
                    prior_pred = self.prior_adapter(prior_pred)

                prior_pred_norm = prior_pred.norm(dim=-1, keepdim=True)
                prior_pred_dir = prior_pred / (prior_pred_norm + eps)

                prior_pred_cos = torch.linalg.vecdot(
                    prior_pred_dir, target_latent_dir, dim=-1
                )

                prior_sim_loss = (
                    1 - prior_pred_cos
                ).mean() * self.config.prior_sim_loss_factor

                losses.update(
                    {
                        "prior_loss": prior_loss,
                        "prior_sim_loss": prior_sim_loss,
                    }
                )
                with torch.no_grad():
                    metrics.update(
                        {
                            "target_latent_len": target_latent_norm.detach().mean().cpu(),
                            "prior_pred_len": prior_pred_norm.detach().mean().cpu(),
                            "prior_l2": torch.norm(
                                prior_pred.detach() - target_latent.detach(), p=2, dim=-1
                            )
                            .mean()
                            .cpu(),
                            "prior_pred_cos": prior_pred_cos.detach().mean().cpu(),
                        }
                    )

                    outputs.update({"prior_pred": prior_pred})

        loss = torch.stack(list(losses.values())).sum()
        losses["loss"] = loss

        for metric_name, metric_value in it.chain(losses.items(), metrics.items()):
            name = f"{stage_prefix}{metric_name}"
            self.log(
                name,
                metric_value,
                prog_bar=name in self.config.prog_bar_metrics,
                on_step=on_step,
                on_epoch=not on_step,
            )

        return loss, outputs, metrics

    @torch.no_grad()
    def evaluate_reconstructions(
        self,
        batch,
        stage: Literal["val", "test"],
        log_images: bool = True,
        num_reconstructions: int = 5,
    ):
        recon_imgs, recon_target = self.get_reconstructions(
            batch, stage, num_reconstructions=num_reconstructions
        )

        stage_prefix = f"{stage.upper()}__"

        if log_images:
            wandb_logger = self.get_wandb_logger()
            if wandb_logger is not None:
                if recon_imgs is not None:
                    wandb_logger.log_image(
                        key=f"{stage_prefix}recon",
                        images=[recon.detach().cpu().float() for recon in recon_imgs],
                    )
                if recon_target is not None:
                    wandb_logger.log_image(
                        key=f"{stage_prefix}recon_target",
                        images=[img.detach().cpu().float() for img in recon_target],
                    )

        if recon_imgs is None or recon_target is None:
            return

        image_paths = [Path(path) for path in batch["img_path"][:num_reconstructions]]
        images = batch_load_images(image_paths).to(
            recon_imgs.device, dtype=recon_imgs.dtype
        )

        lpips_score = self._get_lpips_score(recon_imgs, images)
        self.log(
            f"{stage_prefix}recon_lpips",
            lpips_score,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

    @torch.no_grad()
    def _get_lpips_score(
        self, reconstructions: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")
        lpips.requires_grad_(False)
        lpips.to(self.device)

        # Prepare images for lpips - Need to be in the [-1, 1] range and same shape as reconstructions
        reconstructions = reconstructions * 2 - 1
        images = images / 255.0
        images = torch.nn.functional.interpolate(
            images, reconstructions.shape[-2:], mode="bicubic"
        )
        images = torch.clamp(images, min=0, max=1)
        images = images * 2 - 1

        return lpips(reconstructions, images)

    def get_wandb_logger(self) -> WandbLogger | None:
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    @torch.no_grad()
    def get_reconstructions(
        self,
        batch,
        stage: Literal["val", "test"],
        num_reconstructions: int = 5,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # Imgs Reconstructed: Conditioning on Aligned Brain Latent, Predicting Target Latent with Prior
        # Target Latent: Conditioning on Target Latent, Predicting Target Latent with Prior (Does prior do anything with target?)
        # Target Imgs: Conditioning on Target Latent, Skipping prior, what does perfect reconstruction look like?

        if self.prior is None:
            return None, None

        batch_size = num_reconstructions

        device, dtype = self._get_device_dtype()

        target_latent = batch["high_recon_image_latent"][:batch_size].to(
            device, dtype=dtype
        )
        prior_pred = batch["prior_pred"][:batch_size].to(device, dtype=dtype)
        conditioning = torch.cat([prior_pred, target_latent], dim=0)

        pipe = ReconstructionPipeline.from_stable_diffusion(dtype=dtype, device=device)
        reconstruction = pipe.reconstruct_latents(conditioning, progress_bar=True)
        del pipe

        recon_imgs, recon_target = torch.chunk(reconstruction, 2, dim=0)
        return recon_imgs, recon_target

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
        stage: Literal["train", "val", "test"],
    ) -> DataBatchT:
        img_paths = batch["img_path"]
        eeg_data = batch["eeg_data"]
        device = eeg_data.device

        if stage == "val":
            stage = "test"

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

        return cast(DataBatchT, batch)