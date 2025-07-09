import logging
import os
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from brain_image.configs import BaseConfig
from brain_image.trainer import NICETrainer, NICETrainerConfig
from brain_image.model import EEGEncoderConfig, NICEConfig
from brain_image.data import EEGDatasetConfig

import torch
from pathlib import Path

from brain_image.configs import GlobalConfig

# Initialize wandb login at script startup
try:
    import wandb

    if "WANDB_API_KEY" in os.environ:
        logging.info("WANDB_API_KEY found, attempting to login to wandb...")
        wandb.login(key=os.environ["WANDB_API_KEY"])
        logging.info("Successfully logged in to wandb")
    else:
        logging.warning("WANDB_API_KEY not found in environment")
except ImportError:
    logging.warning("wandb not available")
except Exception as e:
    logging.warning(f"Failed to login to wandb: {e}")


class TrainNICEConfig(BaseConfig):
    """Configuration for NICE training script - handles component composition."""

    script_name: str = "train_nice"
    description: str = "Training script for NICE model with EEG data"

    # Component composition - these will be populated by Hydra
    dataset: EEGDatasetConfig = EEGDatasetConfig()
    model: NICEConfig = NICEConfig(model_name="aligned_synclr")
    trainer: NICETrainerConfig = NICETrainerConfig()
    encoder: EEGEncoderConfig = EEGEncoderConfig()

    # Script-specific settings (not training parameters)
    checkpoint_path: str | None = None
    resume_training: bool = False


def train_nice(trainer: NICETrainer, checkpoint_path: Path | None = None):
    """Clean training function that takes a configured trainer."""

    logging.info(f"Training with configs")
    for key, value in trainer.config.model_dump(mode="json").items():
        logging.info(f"  {key}: {value}")

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

    config = TrainNICEConfig.from_hydra_config(cfg)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Training with config:")
    for key, value in config.model_dump(mode="json").items():
        logging.info(f"  {key}: {value}")

    # Set torch precision
    torch.set_float32_matmul_precision("high")

    # Create trainer with composed components
    trainer = NICETrainer(
        config=config.trainer,
        model_config=config.model,
        dataset_config=config.dataset,
        encoder=config.encoder,
    )

    # Get checkpoint path if specified
    checkpoint_path = None
    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)

    model, test_metrics = train_nice(trainer, checkpoint_path)
    logging.info(f"Finished training with test metrics:")
    for key, value in test_metrics.items():
        logging.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
