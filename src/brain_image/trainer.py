from __future__ import annotations
import datetime
import os
from pathlib import Path
import logging
from typing import Any, Optional, Dict, List, Literal
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, Logger, WandbLogger, CSVLogger
from brain_image.data import EEGDatasetConfig
from brain_image.configs import BaseConfig
from brain_image.model.eeg_alignment import EEGAlignmentConfig, EEGAlignmentModel
from brain_image.utils import get_dtype, init_wandb

class WandbConfig(BaseConfig):
    enabled: bool = True
    project: str = "brain-image"
    entity: Optional[str] = None
    log_model: bool = False
    wandb_tags: List[str] = []
    mode: Literal["online", "offline"] = "online"


class TrainConfig(BaseConfig):
    run_name: str

    compile_model: bool = True
    init_weights: bool = True
    debug_mode: bool = False

    log_dir: Path = Path("logs")
    enable_barebones: bool = False
    checkpoint_monitor: str = "val/loss"
    checkpoint_monitor_mode: Literal["min", "max"] = "max"
    checkpoint_monitor_early_stop: int = 10

    overfit_batches: int = 0
    dtype: Literal["float16", "float32"] = "float32"

    val_check_interval: float = 1.0
    log_every_n_steps: int = 100
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    save_checkpoints: bool = True
    save_top_k: int = 1

    wandb: WandbConfig = WandbConfig()

    accelerator: str | None = None


class EEGAlignTrainerConfig(TrainConfig):
    run_name: str = "eeg_alignment"

    compile_model: bool = True
    init_weights: bool = False
    cache_dir: Path = Path("tensorcache")

    checkpoint_monitor: str = "val_loss"
    checkpoint_monitor_mode: Literal["min", "max"] = "min"


class Trainer:
    def __init__(self, config: TrainConfig, model: EEGAlignmentModel):
        self.config = config
        self.model: EEGAlignmentModel = model
        self.pl_trainer = self.create_pl_trainer()

    def get_tags(self):
        wandb_tags = ["train", *self.config.wandb.wandb_tags]

        if "SLURM_JOB_ID" in os.environ:
            wandb_tags.append("slurm")
        else:
            wandb_tags.append("local")

        if self.config.overfit_batches != 0:
            wandb_tags.append("overfit")

        if self.config.debug_mode:
            wandb_tags.append("debug")

        return wandb_tags

    def create_pl_trainer(self) -> pl.Trainer:
        callbacks: list[pl.Callback] = []
        loggers: list[Logger] = []

        if self.config.save_checkpoints:
            filename = (
                self.config.run_name
                + "-epoch_{epoch:02d}-"
                + self.config.checkpoint_monitor.replace("/", "-")
                + "_{"
                + self.config.checkpoint_monitor
                + ":.4f}"
            )

            checkpoint_callback = ModelCheckpoint(
                monitor=self.config.checkpoint_monitor,
                filename=filename,
                save_top_k=self.config.save_top_k,
                mode=self.config.checkpoint_monitor_mode,
                save_last=True,
                verbose=True,
                auto_insert_metric_name=False
            )
            callbacks.append(checkpoint_callback)

        if self.config.checkpoint_monitor_early_stop > 0:
            early_stopping_callback = EarlyStopping(
                monitor=self.config.checkpoint_monitor,
                patience=self.config.checkpoint_monitor_early_stop,
                mode=self.config.checkpoint_monitor_mode,
                verbose=True,
            )
            callbacks.append(early_stopping_callback)

        tags = sorted(self.get_tags())

        log_path = self.config.log_dir / "-".join(tags)
        log_path.mkdir(parents=True, exist_ok=True)

        loggers.append(
            CSVLogger(
                save_dir=self.config.log_dir,
                name="-".join(tags),
            )
        )
        loggers.append(
            TensorBoardLogger(
                save_dir=self.config.log_dir,
                name="-".join(tags),
                default_hp_metric=False,
            )
        )

        if self.config.wandb.enabled:
            init_wandb()

            name = self.get_train_title()
            wandb_logger = WandbLogger(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=name,
                log_model=self.config.wandb.log_model,
                tags=tags,
                offline=self.config.wandb.mode == "offline",
            )
            loggers.append(wandb_logger)

        precision = "bf16-mixed" if self.config.dtype == "float16" else "32-true"
        accelerator = self.config.accelerator or "auto"

        return pl.Trainer(
            max_epochs=self.model.config.max_epochs,
            callbacks=callbacks if not self.config.enable_barebones else None,
            logger=loggers if not self.config.enable_barebones else [],
            enable_checkpointing=(not self.config.enable_barebones)
            and self.config.save_checkpoints,
            enable_model_summary=not self.config.enable_barebones,
            enable_progress_bar=not self.config.enable_barebones,
            overfit_batches=self.config.overfit_batches,
            precision=precision,
            log_every_n_steps=self.config.log_every_n_steps,
            val_check_interval=self.config.val_check_interval,
            accelerator=accelerator,
        )

    def get_train_title_components(self) -> list[str]:
        components = [
            f"{self.config.run_name}",
            datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
        ]
        if (slurm_job_id := os.environ.get("SLURM_JOB_ID")) is not None:
            components.append(f"slurm_{slurm_job_id}")
        if (slurm_array_job_id := os.environ.get("SLURM_ARRAY_JOB_ID")) is not None:
            components.append(f"array_{slurm_array_job_id}")
        return components

    def get_train_title(self) -> str:
        return "-".join(self.get_train_title_components())

    def train(self, ckpt_path: Optional[Path] = None):
        logging.info(
            f"Starting {self.get_train_title_components()} training with Lightning..."
        )

        ckpt_path_str = str(ckpt_path) if ckpt_path else None

        self.pl_trainer.fit(
            model=self.model,
            ckpt_path=ckpt_path_str,
        )

        logging.info("Training completed!")

    def test(self) -> Dict[str, float]:
        logging.info("Running model testing...")

        results = self.pl_trainer.test(model=self.model)

        if results and len(results) > 0:
            test_metrics = dict(results[0])
            logging.info(f"Test Results: {test_metrics}")
            return test_metrics
        else:
            logging.warning("No test results returned")
            return {}

    def validate(self) -> Dict[str, float]:
        logging.info("Running model validation...")

        results = self.pl_trainer.validate(model=self.model)

        if results and len(results) > 0:
            val_metrics = dict(results[0])
            logging.info(f"Validation Results: {val_metrics}")
            return val_metrics
        else:
            logging.warning("No validation results returned")
            return {}

    def predict(self, dataloader: Optional[DataLoader] = None) -> List[Any]:
        logging.info("Running model predictions...")

        if dataloader is None:
            dataloader = self.model.test_dataloader()

        predictions = self.pl_trainer.predict(
            model=self.model,
            dataloaders=dataloader,
        )

        return predictions or []

    def save_checkpoint(self, filepath: Path):
        self.pl_trainer.save_checkpoint(str(filepath))
        logging.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: Path):
        if self.config.accelerator is None or self.config.accelerator == "auto":
            map_location = "cpu"
        elif self.config.accelerator in ["cpu", "gpu", "cuda"]:
            map_location = "cuda"
        else:  # Assume it's already a valid device string (e.g., "cuda:0")
            map_location = self.config.accelerator

        checkpoint = torch.load(filepath, map_location=map_location)

        self.model.load_state_dict(checkpoint["state_dict"])

        logging.info(f"Loaded checkpoint from {filepath}")


class EEGAlignTrainer(Trainer):
    def __init__(self, trainer_config: EEGAlignTrainerConfig, model: EEGAlignmentModel):
        if isinstance(trainer_config, dict):
            trainer_config = EEGAlignTrainerConfig.model_validate(trainer_config)

        super().__init__(trainer_config, model)
        self.model = model

    def get_tags(self):
        tags = super().get_tags()

        if self.model.config.do_align:
            tags.append("align")
        if self.model.config.do_recon:
            tags.append("recon")
        if self.model.config.do_recon_low:
            tags.append("lowrec")

        tags.append(self.model.config.align_img_encoder)
        tags.append(self.model.config.eeg_encoder)

        return tags

    def get_train_title_components(self) -> list[str]:
        components = super().get_train_title_components()

        tags = self.get_tags()
        components.extend(tags)

        return components
