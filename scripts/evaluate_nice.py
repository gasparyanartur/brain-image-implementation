import json
import logging
from pathlib import Path
from typing import Any, Literal

import hydra
from lightning import Trainer
import torch
import tqdm
from configs import BaseConfig
from model import NICEModel
from data import EEGDatasetConfig

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


def move_all_to_cpu(items: dict[str, Any]) -> dict[str, Any]:
    """Detach all tensors in a dictionary."""
    for key, value in items.items():
        if isinstance(value, torch.Tensor):
            items[key] = value.detach().cpu()
        elif isinstance(value, dict):
            items[key] = move_all_to_cpu(value)
    return items


class EvaluateNiceConfig(BaseConfig):
    config_tag: str = "eval_nice"

    checkpoint_path: Path = Path("models")
    output_path: Path = Path("test_outputs")
    dtype: str = "float32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: Literal[16, 32, 64] = 16


def evaluate_nice(
    config: EvaluateNiceConfig,
):
    device = torch.device(config.device)
    dtype = torch.float32 if config.dtype == "float32" else torch.float16

    # Load the NICE model
    logging.info(f"Loading model from {config.checkpoint_path}")
    model = NICEModel.load_from_checkpoint(
        config.checkpoint_path,
        map_location=device,
        dtype=dtype,
    )
    logging.info(f"Model config: {model.config}")

    model.to(device=device, dtype=dtype)
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
        accelerator=config.device,
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
    config_path="../configs",
    config_name="eval_nice",
    version_base=None,
)
def main(cfg: EvaluateNiceConfig):
    config = EvaluateNiceConfig.from_hydra_config(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Evaluating NICE model with config: {config}")

    evaluate_nice(config)


if __name__ == "__main__":
    main()
