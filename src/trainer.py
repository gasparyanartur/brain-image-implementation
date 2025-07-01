from __future__ import annotations
from abc import abstractmethod
from pathlib import Path
import logging
from typing import Any, Optional, Dict, List, Literal
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, Logger
from src.data import EEGDatasetConfig
from src.configs import BaseConfig
from src.model import NICEModel, NICEConfig


class TrainConfig(BaseConfig):
    num_epochs: int
    learning_rate: float
    batch_size: int
    num_workers: int
    pin_memory: bool
    log_dir: Path = Path("logs")
    enable_barebones: bool = False


class NICETrainerConfig(TrainConfig):
    # Required fields from TrainConfig
    num_epochs: int = 100
    learning_rate: float = 8e-3
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True

    # Model configuration
    submodel_config: NICEConfig
    dataset_config: EEGDatasetConfig = EEGDatasetConfig()

    # Training settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = True
    init_weights: bool = True
    save_checkpoints: bool = True
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    precision: Literal[16, 32, 64] = 32
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    log_every_n_steps: int = 50

    def __init__(self, **kwargs):
        # Allow overriding log_dir and checkpoint_dir
        log_dir = kwargs.pop("log_dir", None)
        checkpoint_dir = kwargs.pop("checkpoint_dir", None)
        model_config = kwargs.get("model_config", None)
        dataset_config = kwargs.get("dataset_config", None)
        from src.model import NICEConfig
        from src.data import EEGDatasetConfig

        if isinstance(model_config, dict):
            kwargs["model_config"] = NICEConfig(**model_config)
        if isinstance(dataset_config, dict):
            kwargs["dataset_config"] = EEGDatasetConfig(**dataset_config)
        super().__init__(**kwargs)
        if log_dir is not None:
            self.log_dir = Path(log_dir)
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        enable_progress_bar = kwargs.pop("enable_progress_bar", None)
        enable_model_summary = kwargs.pop("enable_model_summary", None)
        log_every_n_steps = kwargs.pop("log_every_n_steps", None)
        if enable_progress_bar is not None:
            self.enable_progress_bar = enable_progress_bar
        if enable_model_summary is not None:
            self.enable_model_summary = enable_model_summary
        if log_every_n_steps is not None:
            self.log_every_n_steps = log_every_n_steps

    def create_trainer(self) -> NICETrainer:
        return NICETrainer(self)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def create_pl_trainer(self) -> pl.Trainer:
        callbacks: list[pl.Callback] = []
        loggers: list[Logger] = []

        callbacks.append(
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=None)
        )
        loggers.append(TensorBoardLogger(self.config.log_dir))

        return pl.Trainer(
            max_epochs=self.config.num_epochs,
            callbacks=callbacks if not self.config.enable_barebones else None,
            logger=loggers if not self.config.enable_barebones else [],
            enable_checkpointing=not self.config.enable_barebones,
            enable_model_summary=not self.config.enable_barebones,
            enable_progress_bar=not self.config.enable_barebones,
        )


class NICETrainer(Trainer):
    def __init__(self, config: NICETrainerConfig):
        super().__init__(config)
        self.config = config

        # Initialize Lightning model
        self.model = NICEModel(
            config=self.config.submodel_config,
            dataset_config=self.config.dataset_config,
            compile=self.config.compile_model,
            init_weights=self.config.init_weights,
        )

        # Create Lightning trainer
        self.pl_trainer = self._create_lightning_trainer()

        # Create checkpoint directory
        if self.config.save_checkpoints:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_lightning_trainer(self) -> pl.Trainer:
        """Create Lightning trainer with appropriate callbacks and loggers."""
        callbacks = []
        loggers = []

        # Model checkpoint callback
        if self.config.save_checkpoints:
            checkpoint_callback = ModelCheckpoint(
                monitor="val/loss",
                dirpath=self.config.checkpoint_dir,
                filename="nice-{epoch:02d}-{val/loss:.4f}",
                save_top_k=3,
                mode="min",
                save_last=True,
            )
            callbacks.append(checkpoint_callback)

        # TensorBoard logger
        logger = TensorBoardLogger(
            save_dir=self.config.log_dir,
            name="nice",
            default_hp_metric=False,
        )
        loggers.append(logger)

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.num_epochs,
            callbacks=callbacks,
            logger=loggers,
            enable_progress_bar=self.config.enable_progress_bar,
            enable_model_summary=self.config.enable_model_summary,
            accelerator="auto" if self.config.device == "cuda" else "cpu",
            precision=self.config.precision,
            devices="auto" if self.config.device == "cuda" else 1,
            log_every_n_steps=self.config.log_every_n_steps,
            val_check_interval=0.25,  # Validate every 25% of training epoch
        )

        return trainer

    def train(self, ckpt_path: Optional[Path] = None):
        """Train the model using Lightning."""
        logging.info("Starting NICE training with Lightning...")

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

    def get_model(self) -> NICEModel:
        """Get the trained model."""
        return self.model
