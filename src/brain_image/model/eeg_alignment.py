import datetime
import json
import lightning as pl
import matplotlib.pyplot as plt
from pydantic import Field
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image
from lightning.pytorch.loggers import WandbLogger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.transforms import v2 as tv2
import itertools as it
from contextlib import nullcontext
from pathlib import Path
from brain_image.configs import get_device
from brain_image.data.datamodule import EEGDataModule
from brain_image.data.dataset.eeg_dataset import EEGDatasetConfig
from brain_image.data.io import batch_load_images
from brain_image.data.tensorcache import TensorCache
from brain_image.data.data import (
    LatentTypeMapT,
)
from brain_image.metrics import (
    MetricName,
    evaluate_metrics,
    get_cosine_sim,
    get_retrieval_accuracy,
    get_retrieval_accuracy_with_idx,
    get_top1_acc,
)
from brain_image.model.encoder.eeg_encoder.eeg_encoder import (
    batch_encode_eeg_latent,
    encode_eeg_latent,
)
from brain_image.model.encoder.eeg_encoder.union import (
    EEGEncoderConfigType,
    create_eeg_encoder,
)
from brain_image.model.encoder.img_encoder.union import (
    VAE_ENCODER,
)
from brain_image.model.encoder.img_encoder.union import (
    IMAGE_ENCODER_DIM,
    ImageEncoderName,
)
from brain_image.model.loss import CLIPLoss, InfoNCELoss, SigLipLoss

from brain_image.model.model import (
    TrainingModule,
    TrainingModuleConfig,
)
from brain_image.model.prior import (
    BaseDiffusionPrior,
    DiffusionPriorConfig,
    SimpleDiffusionPrior,
)


from typing import Any, Literal, Mapping, TypedDict, cast
import logging
from brain_image.optimizer import OptimizerConfig, get_optimizer_options
from brain_image.reconstruction import (
    IPAdapterReconstructionPipeline,
    get_batched_reconstructions_from_eeg,
    get_reconstructions,
)
from brain_image.utils import (
    DTYPE,
    VCLR,
    current_fig_to_img,
    find_duplicates,
    gather_dataloader,
    gather_records,
    get_device_from_module,
    get_dtype,
    get_mean_gradients,
    plot_projected_latents,
    reverse_l2_scale,
    reverse_z_scale,
)

import tqdm
import re
import tempfile
import time
import torch


class EEGAlignmentConfig(TrainingModuleConfig):
    align_img_encoder: ImageEncoderName = "unaligned_synclr_vitb16"
    low_level_encoder: VAE_ENCODER = "ip_sdxl_turbo"
    prior_img_encoder: ImageEncoderName = "clip_vitl14"

    eeg_encoder: EEGEncoderConfigType = Field(discriminator="eeg_encoder")

    do_align: bool = True
    do_recon_low: bool = False
    do_recon: bool = True

    align_input_noise: float = 0.0

    plot_lowdim_proj: bool = False
    low_dim_proj_pca: int = 50

    debug_prior_use_target_as_cond: bool = False

    align_loss_type: Literal["clip", "infonce", "siglip"] = "infonce"
    align_loss_epoch: int = 0
    align_loss_factor: float = 0.1
    align_mse_loss_factor: float = 10.0
    prior_align_loss_factor: float = 10.0
    prior_pred_mse_loss_factor: float = 0.01
    prior_sim_mse_loss_factor: float = 1.0

    full_eval_every_epochs: int = 1
    skip_eval_first_epoch: bool = True

    num_reconstructions: int = 3
    clip_temperature: float = 0.07
    clip_bias: float = 0.0
    log_gradients: bool = False
    log_on_step: bool = False

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

    highlighted_val_recons: list[int] = [0, 1, 2]
    highlighted_test_recons: list[int] = [
        161,  # Seaweed
        143,  # Pug
        158,  # Scallop
        166,  # Slide
        65,  # Dreidel
        127,  # Pajamas
        100,  # Jelly Beans
        141,  # Possum
        198,  # Wine
    ]

    max_epochs: int = 100

    seed: int = 42

    prog_bar_metrics: list[str] = [
        "train/loss",
        "val/loss",
        "val/align/mse_loss",
        "val/align/clip_loss",
        "val/prior/loss",
    ]
    test_metrics: list[MetricName] = [
        "pixcorr",
        "ssim",
        "alex2",
        "alex5",
        "inceptionv3",
        "clip",
        "efficientnet",
        "swav",
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
    low_level_latent: torch.Tensor | None
    prior_pred: torch.Tensor | None
    prior_pred_single: torch.Tensor | None


class EEGAlignmentModel(TrainingModule):
    def __init__(
        self,
        config: EEGAlignmentConfig | dict,
        dataset_config: EEGDatasetConfig,
        dtype: torch.dtype = DTYPE,
        init_weights: bool = False,
        compile: bool = True,
        cache_dir: Path = Path("tensorcache"),
        eeg_encoder_path: Path | None = None,
        model_id: str | None = None,
        **kwargs,
    ):

        if isinstance(config, dict):
            config = EEGAlignmentConfig.model_validate(config)

        super().__init__(config, **kwargs)
        self.automatic_optimization = (
            False  # Disable automatic optimization, we will handle it manually
        )

        self.model_id = model_id
        logging.info(f"Seeding everything with seed: {self.config.seed}")
        pl.seed_everything(self.config.seed)

        self.config: EEGAlignmentConfig = (
            config
            if isinstance(config, EEGAlignmentConfig)
            else EEGAlignmentConfig.model_validate(config)
        )

        tensor_cache = TensorCache(cache_dir)
        embeddings_map: LatentTypeMapT = {
            "align_img_latent": (
                self.config.align_img_encoder if self.config.do_align else None
            ),
            "prior_img_latent": (
                self.config.prior_img_encoder if self.config.do_recon else None
            ),
            "low_level_latent": (
                self.config.low_level_encoder if self.config.do_recon_low else None
            ),
            "eeg_latent": None,
        }

        self.data_module = EEGDataModule(
            dataset_config,
            tensor_cache=tensor_cache,
            embeddings_map=embeddings_map,
            embeddings_to_compute_stats=[self.config.prior_img_encoder],
        )

        if init_weights:
            self._init_normal_weights()

        self.prior: BaseDiffusionPrior | None = None
        self.prior_input_encoder: nn.ParameterDict | None = None
        self.prior_input_decoder: nn.ParameterDict | None = None
        self.prior_align_loss: CLIPLoss | None = None

        if self.config.do_recon:
            assert self.config.prior, "Prior config must be provided"

            self.config.prior.d_cond = self.eeg_dim

            self.config.prior.d_input = IMAGE_ENCODER_DIM[self.config.prior_img_encoder]

            emb_stats = cast(
                dict[ImageEncoderName, dict[str, torch.Tensor]] | None,
                {
                    self.config.prior_img_encoder: self.data_module.embedding_stats[
                        "prior_img_latent"
                    ],
                },
            )

            self.prior = SimpleDiffusionPrior(
                self.config.prior,
                latent_name=self.config.prior_img_encoder,
                embedding_stats=emb_stats,
            ).to(dtype)

        elif self.config.do_recon_low:
            raise ValueError(
                "Cannot do low level reconstruction in without high level reconstruction"
            )

        eeg_encoder_path = eeg_encoder_path or self.config.eeg_encoder_path

        self.config.eeg_encoder.d_channels = dataset_config.num_channels
        self.config.eeg_encoder.d_time = dataset_config.time_length
        self.config.eeg_encoder.d_output = self.eeg_dim

        self.eeg_encoder = create_eeg_encoder(
            self.config.eeg_encoder,
            checkpoint_path=eeg_encoder_path,
        )

        self.config.eeg_encoder = cast(
            EEGEncoderConfigType, self.eeg_encoder.config
        )  # Update config with the actual config used, otherwise model dump is wrong

        match self.config.align_loss_type:
            case "clip":
                self.align_loss = CLIPLoss(self.config.clip_temperature)
            case "infonce":
                self.align_loss = InfoNCELoss(self.config.clip_temperature)
            case "siglip":
                self.align_loss = SigLipLoss(
                    self.config.clip_temperature, self.config.clip_bias
                )
            case _:
                raise ValueError(
                    f"Unknown align_loss_type: {self.config.align_loss_type}"
                )

        if compile:
            for module in self.config.modules_to_compile:
                submodule = getattr(self, module)
                if submodule is not None:
                    logging.info(f"Compiling {module}")
                    submodule.compile()

        self.save_hyperparameters(
            {
                "config": self.config.model_dump(mode="json"),
                "dataset_config": self.data_module.config.model_dump(mode="json"),
            },
        )

        self.compile: bool = compile

        self.optimizer_options: list[dict[str, Any]] = []

        self.atleast_one_training_step: bool = False

        logging.info(f"Finished initializing model")

    def get_name(self, timestamp: bool = False) -> str:
        name_components = []

        if timestamp:
            name_components.append(
                datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
            )

        name_components.append(f"eeg_{self.config.eeg_encoder.eeg_encoder}")

        if self.config.do_align:
            name_components.append(f"alig_{self.config.align_img_encoder}")

        if self.config.do_recon:
            name_components.append(f"reco_{self.config.prior_img_encoder}")

        if self.config.do_recon_low:
            name_components.append(f"relo_{self.config.low_level_encoder}")

        return "-".join(name_components)

    @property
    def eeg_dim(self) -> int:
        return IMAGE_ENCODER_DIM[self.config.align_img_encoder]

    def configure_optimizers(self):
        logging.info("Configuring optimizers")

        optimizer_configs = [
            OptimizerConfig(
                name="eeg_encoder",
                modules=[self.eeg_encoder, self.align_loss],
                lr=self.config.encoder_lr,
                min_lr=self.config.encoder_min_lr,
                warmup_epochs=self.config.encoder_warmup_epochs,
                delay_epochs=self.config.encoder_delay_epochs,
                enabled=self.eeg_encoder is not None,
                lr_scheduler="cosine_anneal",
            ),
            OptimizerConfig(
                name="prior",
                modules=[
                    self.prior,
                    self.prior_align_loss,
                    self.prior_input_encoder,
                    self.prior_input_decoder,
                ],
                lr=self.config.prior_lr,
                min_lr=self.config.prior_min_lr,
                warmup_epochs=self.config.prior_warmup_epochs,
                delay_epochs=self.config.prior_delay_epochs,
                enabled=self.prior is not None,
                lr_scheduler="cosine_anneal",
            ),
        ]

        num_train_batches = self.data_module.get_num_batches("train")
        optimizer_options = get_optimizer_options(
            optimizer_configs,
            max_epochs=self.config.max_epochs,
            num_train_batches=num_train_batches,
            modules_to_optimize=self.config.modules_to_train,
            optimizer_params={
                "betas": self.config.betas,
                "weight_decay": self.config.weight_decay,
            },
        )

        self.optimizer_options = optimizer_options
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
        if not undo_compile:
            return super().load_from_checkpoint(checkpoint_path, **kwargs)

        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
        for opt_option in self.optimizer_options:
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
                # f"train/{metric_name}" if ("/" not in metric_name[:10]) else metric_name
                f"train/{metric_name}"
            )
            self.log(
                name,
                metric_value,
                prog_bar=name in self.config.prog_bar_metrics,
                on_step=self.config.log_on_step,
                on_epoch=not self.config.log_on_step,
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
            prefix = (
                "val/" if metric_name.split("/")[0] not in ("debug", "eval") else ""
            )
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

            test_metrics, test_img_outputs, img_paths, idxs = self.run_full_test()

            self.log_test_output(test_metrics, test_img_outputs, idxs)

            if self.log_dir is not None:
                self.dump_test_output(
                    self.log_dir,
                    test_metrics,
                    test_img_outputs,
                    idxs,
                    img_paths,
                    selected_img_idxs=self.config.highlighted_test_recons,
                )

            # Delete training loader to kill persistent workers
            self.data_module.cleanup()

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

        assert (eeg := batch.get("eeg_data")) is not None, "EEG data is not in batch"
        assert (sub := batch.get("sub")) is not None, "Subject is not in batch"

        eeg_latent = encode_eeg_latent(self.eeg_encoder, eeg, sub)

        losses: dict[str, torch.Tensor] = {}
        metrics = {}
        outputs = {}
        cache = {**eeg_latent}

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

        loss = torch.stack([v for v in losses.values()]).sum()
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
    ) -> None:
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

        align_clip_loss, align_logits = self.align_loss(
            eeg_latent_normed, align_img_latent_normed
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
                        "align/top1": get_top1_acc(align_logits, axis=1).cpu(),
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
    ) -> None:
        assert self.config.prior is not None, "Prior config is not initialized"
        assert self.prior is not None, "Prior is not initialized"
        assert (
            target_latent := batch["prior_img_latent"]
        ) is not None, "Prior image latent is not defined"
        assert (
            eeg_latent_normed := cache["eeg_latent_normed"]
        ) is not None, "Normed eeg latent is not defined"
        target_latent = target_latent.to(device)
        eeg_latent_normed = eeg_latent_normed.to(device)

        target_latent = self.prior.scale_target(target_latent)
        batch_size = target_latent.size(0)

        pred = self.prior.sample(
            target_latent,
            eeg_latent_normed,
            timesteps=torch.randint(
                0,
                self.config.prior.num_training_timesteps,
                size=(batch_size,),
                device=device,
            ),
        )

        pred_sim_loss = (
            1
            - get_cosine_sim(
                pred,
                target_latent,
            )
            * self.config.prior_sim_mse_loss_factor
        )
        pred_mse_loss = (
            F.mse_loss(pred, target_latent) * self.config.prior_pred_mse_loss_factor
        )

        losses.update(
            {
                "prior/pred_sim_loss": pred_sim_loss,
                "prior/pred_mse_loss": pred_mse_loss,
            }
        )

        if stage == "val":
            pred_50 = self.prior.sample(
                target_latent,
                eeg_latent_normed,
                timesteps=(self.prior.config.num_training_timesteps // 2)
                * torch.ones(batch_size, device=device, dtype=torch.int32),
                disable_cond_drop=True,
            )

            metrics.update(
                {
                    "prior/pred_cos": VCLR(
                        torch.linalg.vecdot(
                            F.normalize(target_latent), F.normalize(pred_50), dim=-1
                        )
                    )
                }
            )

    @property
    def is_full_val_epoch(self) -> bool:
        is_not_first_or_flag_set = (self.current_epoch > 0) or (
            not self.config.skip_eval_first_epoch
        )
        is_right_mod = (
            self.current_epoch + 1
        ) % self.config.full_eval_every_epochs == 0
        is_right_val_epoch = is_not_first_or_flag_set and is_right_mod
        return is_right_val_epoch

    @torch.no_grad()
    def run_full_validation(
        self, split: Literal["val", "test"]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not (self.is_full_val_epoch or split == "test"):
            return {}, {}

        metrics = {}
        img_outputs = {}

        data_loader = self.data_module.get_dataloader(split)
        all_data = gather_dataloader(data_loader)

        assert (eeg := all_data.get("eeg_data")) is not None, "EEG data is not in batch"
        assert (sub := all_data.get("sub")) is not None, "Subject data is not in batch"

        eeg_latent = batch_encode_eeg_latent(
            self.eeg_encoder,
            cast(torch.Tensor, eeg),
            cast(torch.Tensor, sub),
            batch_size=self.data_module.config.get_batch_size(split),
            progress_bar=False,
        )

        all_data.update(eeg_latent)

        device, dtype = self._get_device_dtype()

        if self.config.do_align:
            metrics_align, img_outputs_align = self._run_validation_align(
                all_data, device  # type: ignore
            )
            metrics.update(metrics_align)
            img_outputs.update(img_outputs_align)

        if self.config.do_recon:
            metrics_prior, img_outputs_prior = self._run_validation_recon(
                all_data, device, split  # type: ignore
            )
            metrics.update(metrics_prior)
            img_outputs.update(img_outputs_prior)

        if self.config.plot_lowdim_proj:
            metrics_plot, img_outputs_plot = self.plot_lowdim_projection(all_data)  # type: ignore
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

        top1_acc, chosen_idx = get_retrieval_accuracy_with_idx(
            eeg_latent_normed, align_img_latent, norm=True
        )

        target_img_paths = [Path(img_paths[i]) for i in indexes[:3]]
        chosen_img_paths = [Path(img_paths[i]) for i in chosen_idx[:3]]
        chosen_imgs = batch_load_images(chosen_img_paths)

        eeg_align_cos = F.cosine_similarity(align_img_latent, eeg_latent_normed).mean()

        metrics = {
            "eval/align/top1_acc": top1_acc.detach().cpu(),
            "eval/align/eeg_cos": eeg_align_cos.detach().cpu(),
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
        if self.prior is None:
            return {}, {}

        if self.current_epoch == 0 and self.config.skip_eval_first_epoch:
            return {}, {}

        assert (target := all_data.get("prior_img_latent")) is not None
        assert (
            eeg_latent_normed := all_data.get("eeg_latent_normed")
        ) is not None, "EEG latent is not in batch"
        assert (prior_img_latent := all_data.get("prior_img_latent")) is not None

        metrics = {}
        img_outputs = {}

        generator = torch.Generator(device).manual_seed(self.config.seed)
        prior_pred = self.prior.batch_generate(
            eeg_latent_normed.to(device),
            generator=generator,
            batch_size=self.data_module.config.get_batch_size("test"),
        )

        recon_idxs = self.config.highlighted_val_recons
        conditioning = torch.cat(
            [
                prior_pred[recon_idxs].to(device),
                prior_img_latent[recon_idxs].to(device),
            ],
            dim=0,
        )
        recon = get_reconstructions(conditioning, pipe_kwargs={"generator": generator})
        recon_pred, recon_target = recon.chunk(2, dim=0)

        img_outputs = {
            "eval/recon/pred": [x.detach().cpu().float() for x in recon_pred],
            "eval/recon/target": [x.detach().cpu().float() for x in recon_target],
        }

        metrics.update(
            {
                "eval/prior/pred_cos": F.cosine_similarity(prior_pred, target)
                .mean()
                .detach()
                .cpu()
            }
        )

        return metrics, img_outputs

    def run_full_test(self, **kwargs) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
        metrics = {}
        img_outputs = {}

        data_loader = self.data_module.get_dataloader("test")
        all_data = gather_dataloader(data_loader)
        all_data = cast(DataBatchT, all_data)

        device = self.device

        assert (
            img_path := all_data.get("img_path")
        ) is not None, "Image path is not in batch"
        assert (
            idxs := all_data.get("img_path")
        ) is not None, "idx not in batch"
        assert (eeg := all_data.get("eeg_data")) is not None, "EEG data is not in batch"
        assert (sub := all_data.get("sub")) is not None, "Subject data is not in batch"

        eeg_latent = batch_encode_eeg_latent(
            self.eeg_encoder,
            cast(torch.Tensor, eeg),
            cast(torch.Tensor, sub),
            batch_size=self.data_module.config.get_batch_size("test"),
            progress_bar=False,
        )["eeg_latent_normed"].to(device)


        if self.config.do_align:
            assert (
                align_img_latent := all_data.get("align_img_latent")
            ) is not None, "Align image latent is not in batch"

            align_img_latent = align_img_latent.to(device)
            brain_acc, image_acc = get_retrieval_accuracy(
                eeg_latent, align_img_latent, norm=True
            )

            metrics.update(
                {
                    "align/brain_acc": brain_acc,
                    "align/image_acc": image_acc,
                }
            )

        if self.config.do_recon:
            metrics_prior, img_outputs_prior = self.full_reconstruction_evaluation(
                img_path, eeg_latent, device, metric_to_eval=kwargs.get("metrics")
            )

            metrics.update(metrics_prior)
            img_outputs.update(img_outputs_prior)

        return metrics, img_outputs, img_path, idxs

    def _run_test_align(
        self,
        eeg_latent: torch.Tensor,
        img_latent: torch.Tensor,
        device: torch.device,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        metrics, img_outputs = {}, {}

        eeg_latent = eeg_latent.to(device)
        img_latent = img_latent.to(device)

        brain_acc, image_acc = get_retrieval_accuracy(eeg_latent, img_latent, norm=True)

        metrics = {
            "align/brain_acc": brain_acc,
            "align/image_acc": image_acc,
        }

        return metrics, img_outputs

    def full_reconstruction_evaluation(
        self,
        img_path: list[str],
        eeg_latent_normed: torch.Tensor,
        device: torch.device,
        metric_to_eval: list[MetricName] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        if self.prior is None:
            return {}, {}

        metric_to_eval = metric_to_eval or self.config.test_metrics
        img_outputs = {}

        reconstructions = get_batched_reconstructions_from_eeg(
            self.prior,
            eeg_latent_normed.to(device),
            self.data_module.config.get_batch_size("test"),
            self.config.seed,
        )
        targets = batch_load_images(img_path, parallel=True, progressbar=True).to(
            device
        )

        metrics = evaluate_metrics(reconstructions, targets, metrics=metric_to_eval)
        metrics = {f"prior/{k}": v for k, v in metrics.items()}

        resize = tv2.Compose(
            [
                tv2.ToDtype(torch.float32, scale=True),
                tv2.Resize((512, 512)),
            ]
        )
        targets = resize(targets)
        recon = resize(reconstructions)

        img_outputs = {
            "prior/reconstruction": [x.detach().cpu() for x in recon],
            "prior/ground_truth": [x.detach().cpu() for x in targets],
        }

        return metrics, img_outputs

    def plot_lowdim_projection(
        self, all_data: DataBatchT
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        latents = []
        labels = []

        for key in ["eeg_latent", "align_img_latent", "prior_img_latent", "prior_pred"]:
            if key in all_data:
                latents.append(all_data[key])
                labels.append(key)

        plot_image = plot_projected_latents(
            latents, labels, "Low Dim Projection", self.config.low_dim_proj_pca
        )
        metrics = {}
        img_outputs = {
            "plot/lowdim": [plot_image],
        }

        return metrics, img_outputs

    def dump_test_output(
        self,
        output_dir: Path,
        metrics: dict[str, Any],
        imgs: dict[str, Any],
        idxs: list[str],
        img_paths: list[str],
        selected_img_idxs: list[int] | None = None,
    ):
        metrics = {name: value.item() for name, value in metrics.items()}

        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Reconstructions
        if self.config.do_recon:
            reconstructions = imgs["prior/reconstruction"]
            ground_truths = imgs["prior/ground_truth"]
            img_dir = Path(output_dir / "reconstructions")
            img_dir.mkdir(parents=True, exist_ok=True)

            for reconstruction, ground_truth, idx, img_path in zip(
                reconstructions, ground_truths, idxs, img_paths
            ):
                if selected_img_idxs is not None and idx not in selected_img_idxs:
                    continue
                save_image(reconstruction, img_dir / f"{idx}_recon.jpg")
                save_image(ground_truth, img_dir / f"{idx}_recon_gt.jpg")

    def log_test_output(self, metrics, imgs, idxs):
        if imgs and ((wandb_logger := self.get_wandb_logger()) is not None):
            # Log selected handful of images to weights and biases
            selected_img_outputs = {k: [] for k in imgs.keys()}
            for k, v in imgs.items():
                for imgs, idx in zip(v, idxs):
                    if (self.config.highlighted_test_recons is None) or (
                        idx in self.config.highlighted_test_recons
                    ):
                        selected_img_outputs[k].append(imgs)

            for k, v in selected_img_outputs.items():
                logging.info(f"Logging {len(v)} images for {k}")
                wandb_logger.log_image(key="full_test/" + k, images=v)

        if metrics:
            for k, v in metrics.items():
                self.log(
                    f"full_test/{k}",
                    v,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )

    def get_wandb_logger(self) -> WandbLogger | None:
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

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
