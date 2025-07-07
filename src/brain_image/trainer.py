from __future__ import annotations
from pathlib import Path
import logging
from typing import Any, Optional, Dict, List, Literal
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, Logger
from brain_image.data import EEGDatasetConfig
from brain_image.configs import BaseConfig
from brain_image.model import Model, NICEModel, NICEConfig


class TrainConfig(BaseConfig):
    run_name: str
    num_epochs: int
    num_workers: int

    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")

    enable_barebones: bool = False
    overfit_batches: int = 0
    precision: Literal[16, 32, 64] = 32

    val_check_interval: float = 0.25
    log_every_n_steps: int = 50

    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    save_checkpoints: bool = True
    save_top_k: int = 1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator: str = "auto" if torch.cuda.is_available() else "cpu"


class NICETrainerConfig(TrainConfig):
    # Required fields from TrainConfig
    run_name: str = "nice"
    num_epochs: int = 100
    num_workers: int = 8

    compile_model: bool = True
    init_weights: bool = True

    # Model configuration - these will be populated by Hydra composition
    model: NICEConfig = NICEConfig(
        model_name="aligned_synclr",
    )
    dataset: EEGDatasetConfig = EEGDatasetConfig()
    encoder: Any = None  # Will be populated by Hydra

    def create_trainer(self) -> NICETrainer:
        return NICETrainer(self)


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
                monitor="val/loss",
                dirpath=self.config.checkpoint_dir,
                filename=f"{self.config.run_name}-{{epoch:02d}}-{{val/loss:.4f}}",
                save_top_k=self.config.save_top_k,
                mode="min",
                save_last=True,
            )
            callbacks.append(checkpoint_callback)

        loggers.append(
            TensorBoardLogger(
                save_dir=self.config.log_dir,
                name=self.config.run_name,
                default_hp_metric=False,
            )
        )

        return pl.Trainer(
            max_epochs=self.config.num_epochs,
            callbacks=callbacks if not self.config.enable_barebones else None,
            logger=loggers if not self.config.enable_barebones else [],
            enable_checkpointing=not self.config.enable_barebones,
            enable_model_summary=not self.config.enable_barebones,
            enable_progress_bar=not self.config.enable_barebones,
            overfit_batches=self.config.overfit_batches,
            precision=self.config.precision,
            devices="auto" if self.config.device == "cuda" else 1,
            log_every_n_steps=self.config.log_every_n_steps,
            val_check_interval=self.config.val_check_interval,
            accelerator=self.config.accelerator,
        )

    def train(self, ckpt_path: Optional[Path] = None):
        """Train the model using Lightning."""
        logging.info(f"Starting {self.config.run_name} training with Lightning...")

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
    def __init__(self, config: NICETrainerConfig):
        model = NICEModel(
            config=config.model,
            dataset_config=config.dataset,
            compile=config.compile_model,
            init_weights=config.init_weights,
        )
        super().__init__(config, model)
