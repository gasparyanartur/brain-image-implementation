import logging
from pathlib import Path
from typing import Any, Literal

import hydra
from omegaconf import DictConfig
from lightning import Trainer
import torch
from brain_image.configs import BaseConfig, get_device
from brain_image.model import NICEModel

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from brain_image.configs import GlobalConfig


def move_all_to_cpu(items: dict[str, Any]) -> dict[str, Any]:
    """Detach all tensors in a dictionary."""
    for key, value in items.items():
        if isinstance(value, torch.Tensor):
            items[key] = value.detach().cpu()
        elif isinstance(value, dict):
            items[key] = move_all_to_cpu(value)
    return items


class EvaluateNiceConfig(BaseConfig):
    checkpoint_path: Path = Path("models")
    output_path: Path = Path("test_outputs")
    device: str | None = None
    precision: Literal[16, 32, 64] = 32


def evaluate_nice(
    config: EvaluateNiceConfig,
):
    device = torch.device(config.device or get_device())
    # Load the NICE model
    logging.info(f"Loading model from {config.checkpoint_path}")
    model = NICEModel.load_from_checkpoint(
        config.checkpoint_path,
        map_location=device,
    )
    logging.info(f"Model config: {model.config}")

    model.requires_grad_(False)
    model.eval()

    test_loader = model.test_dataloader()

    logger = TensorBoardLogger(
        save_dir=config.output_path,
        name=model.config.model_name,
        default_hp_metric=False,
    )

    csv_logger = CSVLogger(
        save_dir=config.output_path,
        name=model.config.model_name,
    )

    trainer = Trainer(
        accelerator="auto",
        precision=config.precision,
        enable_progress_bar=True,
        logger=[logger, csv_logger],
    )

    logging.info("Evaluating model...")
    trainer.test(
        model=model,
        dataloaders=test_loader,
    )
    logging.info("Evaluation complete.")


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="evaluate_nice",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main function for NICE evaluation with clean configuration."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = EvaluateNiceConfig.from_hydra_config(cfg)
    logging.info(f"Evaluating NICE model with config:")
    for key, value in config.model_dump(mode="json").items():
        logging.info(f"  {key}: {value}")

    evaluate_nice(config)


if __name__ == "__main__":
    main()
