import logging
from typing import Any

import hydra
from omegaconf import DictConfig
from brain_image.configs import BaseConfig
from brain_image.trainer import NICETrainer, NICETrainerConfig

import torch
from pathlib import Path

from brain_image.configs import GlobalConfig


class TrainNICEConfig(BaseConfig):
    """Configuration for NICE training script."""

    script_name: str = "train_nice"
    description: str = "Training script for NICE model with EEG data"

    # These will be populated by Hydra composition
    dataset: Any = None
    model: Any = None
    trainer: Any = None
    encoder: Any = None

    # Global settings
    device: str = "cuda"
    precision: int = 16
    num_workers: int = 4
    compile_model: bool = True
    log_path: str = "logs"
    checkpoint_path: str | None = None


def train_nice(trainer: NICETrainer, checkpoint_path: Path | None = None):
    """Clean training function that takes a configured trainer."""

    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

    # Start training
    trainer.train()

    # Test the model
    logging.info("Running final test...")
    test_metrics = trainer.test()

    return trainer.model, test_metrics


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="train_nice",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main function for NICE training with modular configuration."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set torch precision
    torch.set_float32_matmul_precision("high")

    # Create trainer from the composed configuration
    from brain_image.trainer import NICETrainerConfig

    trainer_config = NICETrainerConfig(**cfg.trainer)
    trainer = trainer_config.create_trainer()

    logging.info(f"NICE Training Configuration:")
    logging.info(f"  Script: {cfg.script_name}")
    logging.info(f"  Description: {cfg.description}")
    logging.info(f"  Dataset: {cfg.dataset.data_path}")
    logging.info(f"  Model: {cfg.model.model_name}")
    logging.info(f"  Trainer: {cfg.trainer.run_name}")
    logging.info(f"  Epochs: {cfg.trainer.num_epochs}")
    logging.info(f"  Device: {cfg.device}")
    logging.info(f"  Precision: {cfg.precision}")

    # Get checkpoint path if specified
    checkpoint_path = None
    if cfg.checkpoint_path:
        checkpoint_path = Path(cfg.checkpoint_path)

    model, test_metrics = train_nice(trainer, checkpoint_path)
    logging.info(f"Finished training with test metrics:")
    for key, value in test_metrics.items():
        logging.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
