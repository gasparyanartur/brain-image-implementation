from typing import Literal
import wandb
from brain_image.metrics import get_metric_clip, get_metric_ssim
from brain_image.model.img_encoder import DREAMSIM_IMAGE_ENCODER, VAE_ENCODER, load_vae_encoder
from brain_image.model.loss import DreamsimLoss
from brain_image.model.model import TrainingModule, TrainingModuleConfig
from brain_image.optimizer import OptimizerConfig, get_optimizer_options
from brain_image.configs import get_device
from brain_image.utils import batchify_operation
import logging
import pytorch_lightning as pl
from brain_image.configs import BaseConfig
from brain_image.model.eeg_encoder import create_eeg_encoder
from brain_image.data import (
    EEGDataModule,
    EEGDatasetConfig,
    EmbeddingsMap,
    TensorCache,
    batch_load_images,
)
from torch import nn
from torch.nn import functional as F
from diffusers.models.autoencoders.vae import Decoder

from lightning.pytorch.loggers import WandbLogger


import torch
from torch import nn


class LowLevelModel(torch.nn.Module):
    def __init__(self, in_dim=1024, h=1024, n_blocks=4, upsample_scale: Literal[4, 8, 16] = 4, latent_size: int = 64):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h, h, bias=False),
                    nn.LayerNorm(h),
                    nn.SiLU(inplace=True),
                    nn.Dropout(0.25),
                )
                for _ in range(n_blocks)
            ]
        )

        self.upsample_scale = upsample_scale
        self.latent_size = latent_size
        self.side = latent_size // upsample_scale

        if upsample_scale == 4:
            in_channel = 64
            up_block_channels = [64, 128, 256]
        elif upsample_scale == 8:
            in_channel = 256
            up_block_channels = [64, 128, 256, 256]
        elif upsample_scale == 16:
            in_channel = 256
            up_block_channels = [64, 128, 256, 256, 256]


        self.lin1 = nn.Linear(h, in_channel*self.side**2, bias=False)
        self.norm = nn.GroupNorm(1, in_channel)

        self.upsampler = Decoder(
            in_channels=in_channel,
            out_channels=4,
            up_block_types=tuple("UpDecoderBlock2D" for _ in up_block_channels),
            block_out_channels=tuple(up_block_channels),
            layers_per_block=1,
        )

        self.maps_projector = nn.Identity()

    def forward(self, x, return_transformer_feats=False):
        x = self.lin0(x)
        residual = x

        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, self.side, self.side).contiguous())
        if return_transformer_feats:
            return self.upsampler.decode(x), self.maps_projector(x).flatten(2).permute(
                0, 2, 1
            )
        return self.upsampler(x)


class LowLevelConfig(TrainingModuleConfig):
    encoder_lr: float = 3e-4
    encoder_min_lr: float = 1e-5
    low_level_lr: float = 3e-4
    low_level_min_lr: float = 1e-5
    lr_warmup_epochs: int = 1
    seed: int = 42
    eeg_encoder: str = "atms"
    vae_encoder: VAE_ENCODER = "ip_sdxl_turbo"

    eval_batch_size: int = 32

    metric_log_epochs: int = 5
    highlighted_recons: list[int] | None = [
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

    l2_loss_factor: float = 1.0
    perceptual_loss_factor: float = 1.0

    use_perceptual_loss: bool = False
    perceptual_loss_model: DREAMSIM_IMAGE_ENCODER = "synclr_vitb16"


class LowLevelModule(TrainingModule):
    def __init__(
        self,
        config: LowLevelConfig,
        dataset_config: EEGDatasetConfig = EEGDatasetConfig(subs=[8]),
        compile: bool = False,
    ):
        super().__init__(config)
        self.automatic_optimization = False

        self.config = config
        tensorcache = TensorCache()
        emb_map: EmbeddingsMap = {
            "low_level_latent": self.config.vae_encoder,
            "align_img_latent": None,
            "prior_img_latent": None,
            "prior_img_latent_2": None,
        }
        logging.info(f"Seeding everything with seed: {self.config.seed}")
        pl.seed_everything(self.config.seed)

        self.eeg_dim: int = 1024

        self.data_module = EEGDataModule(
            dataset_config,
            tensorcache,
            embeddings_map=emb_map,
            embeddings_to_compute_stats=[],
        )

        self.eeg_encoder = create_eeg_encoder(
            self.config.eeg_encoder,
            output_dim=self.eeg_dim,
        )

        if self.config.vae_encoder == "ip_sdxl_turbo":
            self.image_size = 512
        elif self.config.vae_encoder == "ip_sdxl_turbo_256":
            self.image_size = 256
        else:
            raise NotImplementedError

        self.latent_size = self.image_size // 8
        self.low_level_encoder = LowLevelModel(in_dim=self.eeg_dim, latent_size=self.latent_size)

        if self.config.perceptual_loss_model is not None and self.config.use_perceptual_loss:
            self.perceptual_loss = DreamsimLoss(self.config.perceptual_loss_model)
        else:
            self.perceptual_loss = None

        self.vae_encoder = load_vae_encoder(self.config.vae_encoder)

        if compile:
            self.eeg_encoder.compile()
            self.low_level_encoder.compile()
            if self.perceptual_loss is not None:
                self.perceptual_loss.compile()

        self.save_hyperparameters(
            {
                "config": self.config.model_dump(mode="json"),
                "dataset_config": self.data_module.config.model_dump(mode="json"),
            },
        )

    def configure_optimizers(self):
        logging.info("Configuring optimizers")

        optimizer_configs = [
            OptimizerConfig(
                name="eeg_encoder",
                modules=[self.eeg_encoder],
                lr=self.config.encoder_lr,
                min_lr=self.config.encoder_min_lr,
                warmup_epochs=self.config.lr_warmup_epochs,
                delay_epochs=0,
                enabled=self.eeg_encoder is not None,
                lr_scheduler="cosine_anneal",
            ),
            OptimizerConfig(
                name="low_level",
                modules=[self.low_level_encoder],
                lr=self.config.low_level_lr,
                min_lr=self.config.low_level_min_lr,
                warmup_epochs=self.config.lr_warmup_epochs,
                delay_epochs=0,
                enabled=self.low_level_encoder is not None,
                lr_scheduler="cosine_anneal",
            ),
        ]

        num_train_batches = self.data_module.get_num_batches("train")
        optimizer_options = get_optimizer_options(
            optimizer_configs,
            max_epochs=self.config.max_epochs,
            num_train_batches=num_train_batches,
            modules_to_optimize=None,
            optimizer_params={},
        )

        self.optimizer_options = optimizer_options
        return optimizer_options

    def run_step(self, batch, batch_idx, stage: Literal["train", "val", "test"]):
        device = get_device()

        losses = {}

        eeg_data = batch["eeg_data"].to(device)
        low_level_target = batch["low_level_latent"].to(device)

        eeg_latent = self.eeg_encoder(eeg_data)
        low_level_latent = self.low_level_encoder(eeg_latent)

        l2_loss = F.mse_loss(low_level_latent, low_level_target)
        losses["l2_loss"] = l2_loss * self.config.l2_loss_factor

        if self.config.use_perceptual_loss:
            assert self.perceptual_loss is not None
            img_paths = batch["img_path"]
            target_imgs = batch_load_images(img_paths).to(device)
            pred_imgs = self.vae_encoder.decode(low_level_latent)
            perceptual_loss = self.perceptual_loss(pred_imgs, target_imgs)
            losses["perceptual_loss"] = perceptual_loss * self.config.perceptual_loss_factor

        loss = sum(losses.values())
        self.log(f"{stage}/loss", loss, on_epoch=True)
        for k, v in losses.items():
            self.log(f"{stage}/losses/{k}", v, on_epoch=True)

        return loss


    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for opt in optimizers:
            opt.zero_grad()

        loss = self.run_step(batch, batch_idx, "train")        

        self.manual_backward(loss)

        for opt in optimizers:
            opt.step()

        for scheduler in schedulers:
            if scheduler is None:
                continue

            try:
                scheduler.step()  # type: ignore
            except ZeroDivisionError as e:
                logging.warning(f"Failed to step scheduler: {e}")

        lr_config = {}
        lr_config["lr/step"] = scheduler.last_epoch if scheduler is not None else -1
        for opt_option in self.optimizer_options:
            lr_config["lr/" + opt_option["name"]] = opt_option[
                "lr_scheduler"
            ].get_last_lr()[0]

        for lr_name, lr_value in lr_config.items():
            self.log(lr_name, lr_value, on_epoch=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.run_step(batch, batch_idx, "val")

        if batch_idx == self.data_module.get_num_batches("val") - 1:
            if (
                self.current_epoch > 0
                and (self.current_epoch + 1) % self.config.metric_log_epochs == 0
            ):
                self.run_full_eval("eval")

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.run_step(batch, batch_idx, "test")

        if batch_idx == self.data_module.get_num_batches("test") - 1:
            self.run_full_eval("test")

    @torch.no_grad()
    def run_full_eval(self, stage):
        device = get_device()
        self.eval()

        prefix = "eval" if stage == "eval" else "test"

        dataloader = self.val_dataloader()

        all_latents = []
        all_paths = []
        all_imgs = []
        all_gt_imgs = []

        with torch.no_grad():
            for batch in dataloader:
                img_paths = batch["img_path"]
                eeg_data = batch["eeg_data"].to(device)
                eeg_latent = self.eeg_encoder(eeg_data)
                low_level_latent = self.low_level_encoder(eeg_latent)

                ex_imgs = batchify_operation(self.vae_encoder.decode, low_level_latent, self.config.eval_batch_size)
                gt_imgs = batch_load_images(img_paths)

                all_latents.append(low_level_latent.cpu())
                all_paths.extend(img_paths)
                all_imgs.append(ex_imgs.cpu())
                all_gt_imgs.append(gt_imgs.cpu())

        all_latents = torch.cat(all_latents, dim=0)
        all_imgs = torch.cat(all_imgs, dim=0)
        all_gt_imgs = torch.cat(all_gt_imgs, dim=0)


        dreamsim_loss_aligned = DreamsimLoss("aligned_synclr_vitb16").to(device)
        dreamsim_loss_unaligned = DreamsimLoss("unaligned_synclr_vitb16").to(device)

        with torch.no_grad():
            metrics = {
                f"{prefix}/ssim": get_metric_ssim(
                    all_imgs.to(device), all_gt_imgs.to(device)
                ),
                f"{prefix}/clip": get_metric_clip(all_imgs.to(device), all_gt_imgs.to(device)),
                f"{prefix}/dreamsim_aligned": dreamsim_loss_aligned(all_imgs.to(device), all_gt_imgs.to(device)),
                f"{prefix}/dreamsim_unaligned": dreamsim_loss_unaligned(all_imgs.to(device), all_gt_imgs.to(device)),
            }

        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True)

        if wandb_logger := self.get_wandb_logger():
            selected_idxs = (
                torch.tensor(self.config.highlighted_recons)
                if (self.config.highlighted_recons is not None)
                else torch.arange(10)
            )
            selected_imgs = all_imgs[selected_idxs]
            selected_gt = all_gt_imgs[selected_idxs]

            image_dump = {
                f"{prefix}/recon_pred": [wandb.Image(img) for img in selected_imgs],
                f"{prefix}/recon_gt": [wandb.Image(img) for img in selected_gt],
            }
            for img_name, imgs in image_dump.items():
                wandb_logger.log_image(
                    key="eval/" + img_name, images=imgs, step=self.global_step
                )

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def get_wandb_logger(self) -> WandbLogger | None:
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None
