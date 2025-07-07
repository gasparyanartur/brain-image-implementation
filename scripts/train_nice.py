import logging
from typing import Literal

import hydra
from omegaconf import DictConfig
from src.brain_image.configs import BaseConfig
from src.brain_image.trainer import NICETrainerConfig

import torch
from pathlib import Path


class TrainNICEConfig(BaseConfig):
    trainer_config: NICETrainerConfig = NICETrainerConfig()
    save_top_k: int = 1
    dtype: str = "float32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: Literal[16, 32, 64] = 32

    num_workers: int = 4
    compile_model: bool = True
    log_path: Path = Path("logs")
    checkpoint_path: Path | None = None


def train_nice(
    config: TrainNICEConfig,
):
    torch.set_float32_matmul_precision("high")

    # Create trainer
    trainer = config.trainer_config.create_trainer()

    # Load checkpoint if provided
    if config.checkpoint_path and config.checkpoint_path.exists():
        logging.info(f"Loading checkpoint from {config.checkpoint_path}")
        trainer.load_checkpoint(config.checkpoint_path)

    # Start training
    trainer.train()

    # Test the model
    logging.info("Running final test...")
    test_metrics = trainer.test()

    return trainer.get_model()


@hydra.main(
    config_path="../configs",
    config_name="train_nice",
    version_base=None,
)
def main(cfg: DictConfig):
    config = TrainNICEConfig.from_hydra_config(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Training NICE model with config: {config}")

    train_nice(config)


if __name__ == "__main__":
    main()
