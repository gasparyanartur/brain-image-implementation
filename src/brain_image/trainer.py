from __future__ import annotations
import datetime
import os
from pathlib import Path
import logging
from typing import Any, Optional, Dict, List, Literal
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, Logger, WandbLogger
from brain_image.data import EEGDatasetConfig
from brain_image.configs import BaseConfig
from brain_image.model import EEGEncoderConfig, Model, NICEModel, NICEConfig
from pydantic import field_validator
import wandb


class TrainConfig(BaseConfig):
    """Base training configuration - focused only on training parameters."""

    # Training parameters
    run_name: str
    num_epochs: int

    # Model compilation and initialization
    compile_model: bool = True
    init_weights: bool = True

    # Logging and checkpointing
    log_dir: Path = Path("logs")
    checkpoint_dir: Path | None = None
    enable_barebones: bool = False
    checkpoint_monitor: str = "val/loss"
    checkpoint_monitor_mode: Literal["min", "max"] = "min"
    overfit_batches: int = 0
    precision: Literal[16, 32, 64] = 16
    val_check_interval: float = 1.0
    log_every_n_steps: int = 100
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    save_checkpoints: bool = True
    save_top_k: int = 1

    # Wandb settings
    enable_wandb: bool = True
    wandb_project: str = "brain-image"
    wandb_entity: Optional[str] = None
    wandb_log_model: bool = False
    wandb_tags: List[str] = []
    wandb_mode: Literal["online", "offline"] = "online"

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator: str = "auto" if torch.cuda.is_available() else "cpu"


class NICETrainerConfig(TrainConfig):
    """NICE-specific trainer configuration - training parameters only."""

    # Required fields from TrainConfig
    run_name: str = "nice"
    num_epochs: int = 100

    # NICE-specific training settings
    compile_model: bool = True
    init_weights: bool = True

    checkpoint_monitor: str = "val/top1_acc"
    checkpoint_monitor_mode: Literal["min", "max"] = "max"

    wandb_tags: list[str] = ["nice"]


class Trainer:
    def __init__(self, config: TrainConfig, model: Model):
        self.config = config
        self.model = model
        self.pl_trainer = self.create_pl_trainer()

    def create_pl_trainer(self) -> pl.Trainer:
        callbacks: list[pl.Callback] = []
        loggers: list[Logger] = []

        if self.config.save_checkpoints:
            checkpoint_callback = ModelCheckpoint(
                monitor=self.config.checkpoint_monitor,
                # dirpath=self.config.checkpoint_dir,
                filename=f"{self.config.run_name}-{{epoch:02d}}-{{{self.config.checkpoint_monitor.replace('/', '_')}:.4f}}",
                save_top_k=self.config.save_top_k,
                mode=self.config.checkpoint_monitor_mode,
                save_last=True,
                verbose=True,
            )
            callbacks.append(checkpoint_callback)

        # Add TensorBoard logger
        loggers.append(
            TensorBoardLogger(
                save_dir=self.config.log_dir,
                name=self.config.run_name,
                default_hp_metric=False,
            )
        )

        # Add Wandb logger if enabled
        if self.config.enable_wandb:
            if "WANDB_API_KEY" not in os.environ:
                logging.warning(
                    "WANDB_API_KEY not found in environment variables, attempting login..."
                )
                wandb.login()

            wandb_tags = [
                *self.config.wandb_tags,
                "train",
            ]
            if "SLURM_JOB_ID" in os.environ:
                wandb_tags.append("slurm")
            else:
                wandb_tags.append("local")

            name = self.get_train_title()
            wandb_logger = WandbLogger(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=name,
                log_model=self.config.wandb_log_model,
                tags=wandb_tags,
                offline=self.config.wandb_mode == "offline",
            )
            loggers.append(wandb_logger)

        precision = "bf16-mixed" if self.config.precision == 16 else "32-true"

        return pl.Trainer(
            max_epochs=self.config.num_epochs,
            callbacks=callbacks if not self.config.enable_barebones else None,
            logger=loggers if not self.config.enable_barebones else [],
            enable_checkpointing=not self.config.enable_barebones,
            enable_model_summary=not self.config.enable_barebones,
            enable_progress_bar=not self.config.enable_barebones,
            overfit_batches=self.config.overfit_batches,
            precision=precision,
            devices="auto",
            log_every_n_steps=self.config.log_every_n_steps,
            val_check_interval=self.config.val_check_interval,
            accelerator=self.config.accelerator,
        )

    def get_train_title_components(self) -> list[str]:
        components = [
            f"{self.config.run_name}",
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        ]
        if "SLURM_JOB_ID" in os.environ:
            components.append(os.environ["SLURM_JOB_ID"])
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            components.append(os.environ["SLURM_ARRAY_TASK_ID"])
        return components

    def get_train_title(self) -> str:
        return "-".join(self.get_train_title_components())

    def train(self, ckpt_path: Optional[Path] = None):
        """Train the model using Lightning."""
        logging.info(
            f"Starting {self.get_train_title_components()} training with Lightning..."
        )

        # Convert checkpoint path to string if provided
        ckpt_path_str = str(ckpt_path) if ckpt_path else None

        # Start training
        self.pl_trainer.fit(
            model=self.model,
            ckpt_path=ckpt_path_str,
        )

        logging.info("Training completed!")

    def test(self) -> Dict[str, float]:
        """Test the model using Lightning."""
        logging.info("Running model testing...")

        results = self.pl_trainer.test(model=self.model)

        # Extract metrics from results
        if results and len(results) > 0:
            test_metrics = dict(results[0])
            logging.info(f"Test Results: {test_metrics}")
            return test_metrics
        else:
            logging.warning("No test results returned")
            return {}

    def validate(self) -> Dict[str, float]:
        """Validate the model using Lightning."""
        logging.info("Running model validation...")

        results = self.pl_trainer.validate(model=self.model)

        # Extract metrics from results
        if results and len(results) > 0:
            val_metrics = dict(results[0])
            logging.info(f"Validation Results: {val_metrics}")
            return val_metrics
        else:
            logging.warning("No validation results returned")
            return {}

    def predict(self, dataloader: Optional[DataLoader] = None) -> List[Any]:
        """Run predictions using Lightning."""
        logging.info("Running model predictions...")

        if dataloader is None:
            dataloader = self.model.test_dataloader()

        predictions = self.pl_trainer.predict(
            model=self.model,
            dataloaders=dataloader,
        )

        return predictions or []

    def save_checkpoint(self, filepath: Path):
        """Save model checkpoint manually."""
        self.pl_trainer.save_checkpoint(str(filepath))
        logging.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: Path):
        """Load model from checkpoint."""
        # Load the checkpoint
        checkpoint = torch.load(filepath, map_location=self.config.device)

        # Load model state
        self.model.load_state_dict(checkpoint["state_dict"])

        logging.info(f"Loaded checkpoint from {filepath}")


class NICETrainer(Trainer):
    def __init__(
        self,
        config: NICETrainerConfig,
        model_config: NICEConfig,
        dataset_config: EEGDatasetConfig,
        encoder: Any = None,
    ):
        if isinstance(config, dict):
            config = NICETrainerConfig.model_validate(config)
        if isinstance(model_config, dict):
            model_config = NICEConfig.model_validate(model_config)
        if isinstance(dataset_config, dict):
            dataset_config = EEGDatasetConfig.model_validate(dataset_config)
        if isinstance(encoder, dict):
            encoder = EEGEncoderConfig.model_validate(encoder)

        model = NICEModel(
            config=model_config,
            dataset_config=dataset_config,
            compile=config.compile_model,
            init_weights=config.init_weights,
        )
        self.model_config: NICEConfig = model_config
        super().__init__(config, model)

    def get_train_title_components(self) -> list[str]:
        return super().get_train_title_components() + [self.model_config.model_name]
