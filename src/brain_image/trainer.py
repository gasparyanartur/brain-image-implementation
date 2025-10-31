from __future__ import annotations
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
from brain_image.configs import BaseConfig, get_device_str
from brain_image.model.eeg_alignment import EEGAlignmentConfig, EEGAlignmentModel
from brain_image.utils import create_model_id, get_dtype, init_wandb

class WandbConfig(BaseConfig):
    enabled: bool = True
    project: str = "brain-image"
    entity: Optional[str] = None
    log_model: bool = False
    mode: Literal["online", "offline"] = "online"


class TrainConfig(BaseConfig):
    run_name: str

    compile_model: bool = True
    init_weights: bool = True
    debug_mode: bool = False

    log_dir: Path = Path("logs/train")
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

    make_subdir: bool = False

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

        log_path = self.config.log_dir 
        if self.config.make_subdir:
            log_path = log_path / self.get_train_title()

        log_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Logging to path {log_path}...")

        model_id = create_model_id()
        if (slurm_array_job_id := os.getenv("SLURM_ARRAY_JOB_ID")) is not None:
            model_id += f"-slurmarr{slurm_array_job_id}"
            if (slurm_task_id := os.getenv("SLURM_ARRAY_TASK_ID")) is not None:
                model_id += f"_{slurm_task_id}"

        elif (slurm_job_id := os.getenv("SLURM_JOB_ID")) is not None:
            model_id += f"-slurm{slurm_job_id}"


        loggers.append(
            CSVLogger(
                save_dir=log_path,
                name=model_id,
                version=0
            )
        )
        loggers.append(
            TensorBoardLogger(
                save_dir=log_path,
                name=model_id,
                default_hp_metric=False,
                version=0
            )
        )

        if self.config.wandb.enabled:
            init_wandb()

            wandb_logger = WandbLogger(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=model_id,
                log_model=self.config.wandb.log_model,
                tags=self.get_title_components(),
                offline=self.config.wandb.mode == "offline",
                group=os.environ.get("SLURM_ARRAY_JOB_ID"),
                job_type="train"
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
            enable_progress_bar=(not self.config.enable_barebones) and self.config.enable_progress_bar,
            overfit_batches=self.config.overfit_batches,
            precision=precision,
            log_every_n_steps=self.config.log_every_n_steps if self.model.config.log_on_step else 1_000_000_000,
            val_check_interval=self.config.val_check_interval,
            accelerator=accelerator,
        )

    def get_title_components(self, timestamp=False) -> list[str]:
        components = []

        if self.config.debug_mode:
            components.append("debug")

        if (slurm_job_id := os.environ.get("SLURM_JOB_ID")) is not None:
            components.append(f"slurm")

        if (slurm_task_id := os.environ.get("SLURM_ARRAY_TASK_ID")) is not None:
            components.append(f"slurmarr")

        if slurm_job_id is None and slurm_task_id is None:
            components.append("local")

        return components

    def get_train_title(self, timestamp=False) -> str:
        return "-".join(self.get_title_components(timestamp=timestamp))

    def train(self, ckpt_path: Optional[Path] = None):
        logging.info(
            f"Starting training with Lightning..."
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
        checkpoint = torch.load(filepath, map_location=get_device_str())

        self.model.load_state_dict(checkpoint["state_dict"])

        logging.info(f"Loaded checkpoint from {filepath}")


class EEGAlignTrainer(Trainer):
    def __init__(self, trainer_config: EEGAlignTrainerConfig, model: EEGAlignmentModel):
        if isinstance(trainer_config, dict):
            trainer_config = EEGAlignTrainerConfig.model_validate(trainer_config)

        super().__init__(trainer_config, model)
        self.model = model

