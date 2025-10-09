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
from functools import cached_property
from pathlib import Path
from brain_image.configs import BaseConfig, get_device
from brain_image.data import (
    EEGDataModule,
    EEGDatasetConfig,
    TensorCache,
    batch_load_images,
    get_image_paths,
)
from brain_image.model.eeg_encoder import create_eeg_encoder
from brain_image.model.eeg_encoder.eeg_encoder import EEGEncoder
from brain_image.model.loss import CLIPLoss, InfoNCELoss
from brain_image.model.model import (
    normalize_projection,
)
from brain_image.model.prior import (
    DiffusionPriorConfig,
    SimpleDiffusionPrior,
)


from typing import Any, Literal, Mapping, Sequence, TypedDict, cast
import logging
from brain_image.reconstruction import IPAdapterReconstructionPipeline, ReconstructionPipeline
from brain_image.utils import (
    DTYPE,
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
    align_target_encoder: str = "unaligned_synclr_vitb16"
    recon_latent_encoder: str = "sd_variations_v2"
    prior_img_encoder: str = "clip_vitl14"
    eeg_encoder: str = "nice"

    do_align: bool = True
    do_recon_low: bool = False
    do_recon: bool = True

    align_input_noise: float = 0.0

    plot_lowdim_proj: bool = True
    low_dim_proj_pca: int = 50

    align_loss_type: Literal["clip", "infonce"] = "infonce"
    align_loss_epoch: int = 0
    align_loss_factor: float = 0.1
    align_mse_loss_factor: float = 1
    prior_loss_factor: float = 0.1

    rescale_proj_by_mean: bool = False
    norm_eeg_latent: bool = True

    full_eval_every_epochs: int = 1
    skip_eval_first_epoch: bool = True

    img_latent_dim: int = 768
    project_dim: int = 768

    prior_debug_mode: bool = (
        False  # If True, will use the target image in the prior and disable alignment
    )

    num_reconstructions: int = 3
    temperature_init: float = 0.07
    log_gradients: bool = False

    prior_config: DiffusionPriorConfig | None = DiffusionPriorConfig()

    encoder_lr: float = 3e-4
    prior_lr: float = 3e-4
    align_loss_lr: float = 3e-4

    encoder_min_lr: float = 1e-5
    prior_min_lr: float = 1e-6
    align_loss_min_lr: float = 1e-7

    encoder_warmup_epochs: int = 1
    prior_warmup_epochs: int = 1
    align_loss_warmup_epochs: int = 1

    encoder_delay_epochs: int = 0
    prior_delay_epochs: int = 0
    align_loss_delay_epochs: int = 0

    lr_scheduler: Literal["none", "cosine_anneal"] = "cosine_anneal"
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01

    # freeze_eeg_encoder: bool = False

    max_epochs: int = 100

    warmup_start_frac: float = 0.35
    data_seed: int = 42
    seed: int = 42

    prog_bar_metrics: list[str] = [
        "TRAIN__loss",
        "VAL__loss",
        "VAL__align_top1_acc",
        "VAL__prior_pred_cos",
    ]


class DataBatchT(TypedDict):
    img_path: list[str] | None
    eeg_data: torch.Tensor | None
    sub: torch.Tensor | None
    idx: torch.Tensor | None
    eeg_latent: torch.Tensor | None
    align_img_latent: torch.Tensor | None
    prior_img_latent: torch.Tensor | None
    recon_latent: torch.Tensor | None
    prior_pred: torch.Tensor | None
    prior_pred_single: torch.Tensor | None


class EEGAlignmentModel(pl.LightningModule):
    def __init__(
        self,
        config: EEGAlignmentConfig | dict[str, Any],
        dataset_config: EEGDatasetConfig | dict[str, Any],
        dtype: torch.dtype = DTYPE,
        init_weights: bool = False,
        preload_latents: bool = True,
        compile: bool = True,
        modules_to_compile: list[str] = [
            "eeg_encoder",
            "prior",
        ],
        cache_dir: Path = Path("cache/tensorcache"),
        eeg_encoder_path: Path | None = None,
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
            self.config.do_recon_low = False
            self.config.do_recon = True

        if init_weights:
            self._init_normal_weights()

        self.prior: SimpleDiffusionPrior | None = None
        if self.config.do_recon:
            assert self.config.prior_config, "Prior config must be provided"
            self.prior = SimpleDiffusionPrior(self.config.prior_config).to(dtype)

        elif self.config.do_recon_low:
            raise ValueError(
                "Cannot do low level reconstruction in without high level reconstruction"
            )

        if preload_latents:
            self._preload_latents()

        if self.config.prior_debug_mode:
            self.eeg_encoder = None
        else:
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

            for module in modules_to_compile:
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
                name="prior",
                model=self.prior,
                lr=self.config.prior_lr,
                min_lr=self.config.prior_min_lr,
                warmup_epochs=self.config.prior_warmup_epochs,
                delay_epochs=self.config.prior_delay_epochs,
            ),
            OptimizerConfig(
                name="align_loss",
                model=self.align_loss,
                lr=self.config.align_loss_lr,
                min_lr=self.config.align_loss_min_lr,
                warmup_epochs=self.config.align_loss_warmup_epochs,
                delay_epochs=self.config.align_loss_delay_epochs,
            ),
        ]
        optimizer_configs = [x for x in optimizer_configs if x.model is not None]

        optimizer_options = []
        for optimizer_config in optimizer_configs:
            logging.info(
                f"Creating optimizer: {optimizer_config.name} - lr: {optimizer_config.lr}, min_lr: {optimizer_config.min_lr}, warmup_epochs: {optimizer_config.warmup_epochs}, delay_epochs: {optimizer_config.delay_epochs}"
            )
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
                    path, self.config.align_target_encoder, split
                )
            if self.config.do_recon:
                self._get_image_latent_from_cache(
                    path, self.config.prior_img_encoder, split
                )
            if self.config.do_recon_low:
                self._get_image_latent_from_cache(
                    path, self.config.recon_latent_encoder, split
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
                    ("prior", self.prior),
                ]
                grad_modules_to_log = [
                    x for x in grad_modules_to_log if x[1] is not None
                ]
                for name, module in grad_modules_to_log:
                    grads = get_mean_gradients(module)
                    if grads is not None:
                        self.log(
                            f"GRAD__{name}",
                            grads,
                            prog_bar=False,
                            on_step=True,
                            on_epoch=False,
                        )

        return loss

    def validation_step(
        self, batch, batch_idx
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
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
        loss, outputs, metrics = self._run_step(batch, batch_idx, "test")

        if batch_idx == self.num_test_batches - 1:
            self._run_full_validation(split="test")

        return loss, outputs, metrics

    def get_eeg_latent(
        self,
        batch: DataBatchT,
        device,
        dtype,
        use_align_if_debug: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        if not normalize:
            raise ValueError(f"Non-normalized EEG not supported")

        if use_align_if_debug and self.config.prior_debug_mode:
            assert (
                (align_img_latent_normed := batch.get("align_image_latent_normed")) is not None
            ), "Align image latent is not in batch"
            eeg_latent_normed = align_img_latent_normed.to(device, dtype=dtype)

        else:
            assert self.eeg_encoder is not None, "EEG encoder is not initialized"
            assert (subs := batch.get("sub")) is not None, "Subject is not in batch"
            assert (
                eeg_data := batch.get("eeg_data")
            ) is not None, "EEG data is not in batch"

            eeg_data = eeg_data.to(device, dtype=dtype)
            subs = subs.to(device)
            eeg_latent_normed = self.eeg_encoder(eeg_data, subs)

            eeg_latent_normed = normalize_projection(
                eeg_latent_normed, self.config.rescale_proj_by_mean
            )

        return eeg_latent_normed

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

        device, dtype = self._get_device_dtype()

        if not key_in_dict("eeg_latent", all_data):
            all_data["eeg_latent"] = self.get_all_eeg_latents(
                all_data,
                batch_size=batch_size,
                progress_bar=progress_bar,
                normalize=True,
            )

        assert (
            eeg_latent := all_data.get("eeg_latent")
        ) is not None, "EEG latent is not in batch"

        all_prior_preds_ = []
        with tqdm.tqdm(
            total=eeg_latent.size(0),
            desc="Prior sampling",
            disable=not progress_bar,
        ) as pbar:
            for i in range(0, eeg_latent.size(0), batch_size):
                latent_batch = eeg_latent[i : i + batch_size].to(device)
                prior_pred = self.prior.generate(
                    conditioning=latent_batch,
                    generator=generator,
                    **prior_kwargs,
                )

                all_prior_preds_.append(prior_pred.detach().cpu())
                pbar.update(latent_batch.size(0))

        return torch.cat(all_prior_preds_, dim=0)

    @torch.no_grad()
    def get_all_eeg_latents(
        self,
        all_data: DataBatchT,
        batch_size: int = 512,
        progress_bar: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor | None:
        assert (
            "idx" in all_data and all_data["idx"] is not None
        ), "Index is not in batch"

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
                    },
                )
                eeg_latent = self.get_eeg_latent(
                    batch_data,
                    device,
                    dtype,
                    use_align_if_debug=True,
                    normalize=normalize,
                )
                all_latents.append(eeg_latent.detach().cpu())
                pbar.update(len(eeg_latent))

        return torch.cat(all_latents, dim=0)

    @torch.no_grad()
    def plot_lowdim_projection(
        self, all_data: DataBatchT, show_plot: bool = False
    ) -> dict[str, Any]:
        assert (
            align_img_latent_normed := all_data.get("align_img_latent_normed")
        ) is not None, "Align image latent is not in batch"
        assert (
            eeg_latent_normed := all_data.get("eeg_latent_normed")
        ) is not None, "EEG latent is not in batch"
        if self.config.do_recon:
            assert (
                prior_img_latent := all_data.get("prior_img_latent")
            ) is not None, "Prior image latent is not in batch"
            assert (
                prior_pred := all_data.get("prior_pred")
            ) is not None, "Prior pred is not in batch"

        logging.info(
            f"Projecting latents from dim {prior_img_latent.size(1)} to 2 dimensions"
        )

        n = len(prior_img_latent)

        pca = PCA(n_components=self.config.low_dim_proj_pca)
        tsne = TSNE(n_components=2)

        latents_highdim = [eeg_latent_normed, align_img_latent_normed]
        labels = ["eeg_latent", "align_target_latent"]
        c = ["blue", "red"]
        if self.config.do_recon:
            latents_highdim.extend([prior_img_latent, prior_pred])
            labels.extend(["prior_img_latent", "prior_pred"])
            c.extend(["green", "orange"])
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
            data_loader, lambda b: self.prepare_batch(b, stage=split)  # type: ignore
        )
        all_data = cast(DataBatchT, all_data)

        assert (align_img_latent_normed := all_data.get("align_img_latent_normed")) is not None, "Align image latent is not in batch"
        eeg_latent_normed = self.get_all_eeg_latents(
            all_data,
            batch_size=self.data_module.config.val_batch_size,
            progress_bar=False,
            normalize=True
        )
        assert eeg_latent_normed is not None, "EEG latents are not initialized"
        assert (img_paths := all_data.get("img_path")) is not None, "Image paths are not in batch"
        assert (indexes := all_data.get("idx")) is not None, "Indices are not in batch"

        all_data["eeg_latent_normed"] = eeg_latent_normed
        device, dtype = self._get_device_dtype()

        if self.config.do_align:
            sim = eeg_latent_normed @ align_img_latent_normed.T
            top_sim = sim.topk(1, dim=-1).indices.flatten()  # <B, 1>
            chosen_idx = indexes[top_sim]
            top1_acc = (chosen_idx == indexes).float().mean()

            chosen_img_paths = [img_paths[i] for i in chosen_idx]

            eeg_align_cos = torch.linalg.vecdot(
                eeg_latent_normed, align_img_latent_normed, dim=-1
            )

            img_outputs.update(
                {
                    f"align_test_target": [wandb.Image(x) for x in img_paths[:3]],  # type: ignore
                    f"align_test_chosen": [wandb.Image(x) for x in chosen_img_paths[:3]],  # type: ignore
                }
            )
            metrics.update(
                {
                    "eval_align_top1_acc": top1_acc,
                    "eval_align_eeg_cos": eeg_align_cos.detach().mean().cpu(),
                }
            )

        if self.config.do_recon:
            assert (prior_img_latent := all_data.get("prior_img_latent")) is not None

            gen = torch.Generator(device).manual_seed(self.config.seed)
            all_prior_preds = self.get_all_prior_preds(
                all_data,
                progress_bar=True,
                generator=gen,
                batch_size=self.data_module.config.val_batch_size,
            )
            assert all_prior_preds is not None

            all_data["prior_pred"] = all_prior_preds

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
            prior_img_latent = get_norm_dir_len(prior_img_latent)

            prior_img_latent_cos = torch.linalg.vecdot(
                prior_pred_info.dir, prior_img_latent.dir, dim=-1
            ).mean()
            metrics.update(
                {
                    "eval_prior_recon_cos": prior_img_latent_cos.cpu(),
                    "eval_prior_pred_len_ratio": (
                        prior_img_latent.len / prior_pred_info.len
                    ).cpu(),
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

        eeg_latent_normed = self.get_eeg_latent(
            batch,
            device,
            self.dtype,
            use_align_if_debug=False,
        )

        losses: dict[str, torch.Tensor] = {}
        metrics = {}
        outputs = {}
        cache = {
            "eeg_latent_normed": eeg_latent_normed,
        }

        on_step = stage == "train"

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
                (self.epoch >= self.config.prior_delay_epochs)
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
            align_img_latent_normed := batch.get("align_img_latent_normed")
        ) is not None, "Align image latent is not initialized"
        assert (idx := batch.get("idx")) is not None, "Index is not initialized"
        assert self.align_loss is not None, "Align loss is not initialized"

        align_img_latent_normed = align_img_latent_normed.to(device)

        if stage == "train" and self.config.align_input_noise > 0:
            align_img_latent_normed = align_img_latent_normed + (
                torch.randn_like(align_img_latent_normed)
                * align_img_latent_normed.norm(dim=-1, keepdim=True)
                * self.config.align_input_noise
            )

        with torch.no_grad():
            # There might be duplicates (different subjects, same image)
            labels = (idx.unsqueeze(0) == idx.unsqueeze(1)).float()
            labels = labels.to(device)

        align_clip_loss, align_sim = self.align_loss(
            eeg_latent_normed, align_img_latent_normed, labels=labels
        )

        align_clip_loss = (
            align_clip_loss
            * self.config.align_loss_factor
            * (self.epoch >= self.config.align_loss_epoch)
        )
        align_mse_loss = (
            torch.nn.functional.mse_loss(eeg_latent_normed, align_img_latent_normed)
            * self.config.align_mse_loss_factor
        )
        align_cos = align_sim.diag()

        losses.update(
            {
                "align_mse_loss": align_mse_loss,
                "align_clip_loss": align_clip_loss,
            }
        )

        with torch.no_grad():
            metrics.update(
                {
                    "align_cos": align_cos.mean().detach().cpu(),
                    "align_logit_scale": self.align_loss.logit_scale.detach().cpu(),
                }
            )

            outputs.update({})

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
        assert self.config.prior_config is not None, "Prior config is not initialized"
        assert self.prior is not None, "Prior is not initialized"
        assert (
            target_latent := batch["prior_img_latent"]
        ) is not None, "Prior image latent is not defined"
        assert (
            eeg_latent_normed := cache["eeg_latent_normed"]
        ) is not None, "Normed eeg latent is not defined"

        batch_size = target_latent.size(0)

        scale = math.sqrt(target_latent.size(-1))
        target_latent_scaled = F.normalize(target_latent) * scale

        noise = torch.randn_like(target_latent_scaled)
        timesteps = torch.randint(0, self.config.prior_config.num_training_timesteps, size=(batch_size,), device=device)
        noisy_latent = self.prior.scheduler.add_noise(target_latent_scaled, noise, timesteps=cast(torch.IntTensor, timesteps))

        noise_pred = self.prior.forward(noisy_latent, timesteps, eeg_latent_normed, self.config.prior_config.cond_drop_prob)
        prior_loss = torch.nn.functional.mse_loss(noise_pred, noise) * self.config.prior_loss_factor

        losses.update(
            {
                "prior_loss": prior_loss,
            }
        )

    @torch.no_grad()
    def evaluate_reconstructions(
        self,
        batch,
        stage: Literal["val", "test"],
        log_images: bool = True,
        num_reconstructions: int = 5,
    ):
        recon_pred, recon_target = self.get_reconstructions(
            batch, stage, num_reconstructions=num_reconstructions
        )

        stage_prefix = f"{stage.upper()}__"

        if log_images:
            wandb_logger = self.get_wandb_logger()
            if wandb_logger is not None:
                if recon_pred is not None:
                    wandb_logger.log_image(
                        key=f"{stage_prefix}recon",
                        images=[recon.detach().cpu().float() for recon in recon_pred],
                    )
                if recon_target is not None:
                    wandb_logger.log_image(
                        key=f"{stage_prefix}recon_target",
                        images=[img.detach().cpu().float() for img in recon_target],
                    )

        if recon_pred is None or recon_target is None:
            return

        lpips_score = self.get_lpips_score(recon_pred, recon_target)
        self.log(
            f"{stage_prefix}recon_lpips",
            lpips_score,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

    @torch.no_grad()
    def get_lpips_score(
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
    def get_reconstructions(
        self,
        batch,
        stage: Literal["val", "test"],
        num_reconstructions: int = 5,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # Imgs Reconstructed: Conditioning on Aligned Brain Latent, Predicting Target Latent with Prior
        # Target Latent: Conditioning on Target Latent, Predicting Target Latent with Prior (Does prior do anything with target?)
        # Target Imgs: Conditioning on Target Latent, Skipping prior, what does perfect reconstruction look like?

        if self.prior is None:
            return None, None

        batch_size = num_reconstructions

        device, dtype = self._get_device_dtype()

        target_latent = batch["prior_img_latent"][:batch_size].to(
            device, dtype=dtype
        )
        prior_pred = batch["prior_pred"][:batch_size].to(device, dtype=dtype)
        conditioning = torch.cat([F.normalize(prior_pred), F.normalize(target_latent)], dim=0)

        pipe = IPAdapterReconstructionPipeline.load_pretrained(device=device)
        reconstruction = pipe.reconstruct_latents(conditioning)
        del pipe

        recon_imgs, recon_target = torch.chunk(reconstruction, 2, dim=0)
        return recon_imgs, recon_target

    def _get_image_latent_from_cache(
        self, img_path: Path, *model_config: str
    ) -> torch.Tensor:
        return self.tensor_cache.get(str(img_path), *model_config)

    def _get_batch_from_cache(
        self,
        img_paths: list[Path],
        *model_config: str,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        batch = torch.stack(
            [
                self._get_image_latent_from_cache(img_path, *model_config)
                for img_path in img_paths
            ]
        )
        params = {}
        if dtype is not None:
            params["dtype"] = dtype
        if device is not None:
            params["device"] = device
        if params:
            batch = batch.to(**params)

        return batch

    def prepare_batch(
        self,
        batch: dict[str, Any],
        stage: Literal["train", "val", "test"],
    ) -> DataBatchT:
        img_paths = batch["img_path"]
        eeg_data = batch["eeg_data"]
        device = eeg_data.device
        dtype = eeg_data.dtype

        if stage == "val":
            stage = "test"

        batch["align_img_latent_normed"] = normalize_projection(
            self._get_batch_from_cache(
                img_paths,
                self.config.align_target_encoder,
                stage,
                device=device,
                dtype=dtype,
            ),
            self.config.rescale_proj_by_mean,
        )

        if self.config.do_recon:
            batch["prior_img_latent"] = self._get_batch_from_cache(
                img_paths,
                self.config.prior_img_encoder,
                stage,
                device=device,
                dtype=dtype,
            )


        if self.config.do_recon_low:
            batch["recon_img_latent"] = self._get_batch_from_cache(
                img_paths,
                self.config.recon_latent_encoder,
                stage,
                device=device,
                dtype=dtype,
            )


        return cast(DataBatchT, batch)
