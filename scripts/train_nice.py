import logging
from typing import Literal

import hydra
from omegaconf import DictConfig
from src.configs import BaseConfig
from src.data import EEGDatasetConfig
from src.model import NICEConfig, NICEModel

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


from pathlib import Path

from src.utils import get_dtype


class TrainNICEConfig(BaseConfig):
    nice_config: NICEConfig = NICEConfig()
    dataset_config: EEGDatasetConfig = EEGDatasetConfig()
    save_top_k: int = 1
    dtype: str = "float32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: Literal[16, 32, 64] = 32
    train_val_split: float = 0.9

    num_workers: int = 4
    compile_model: bool = True
    log_path: Path = Path("logs")
    checkpoint_path: Path | None = None


def train_nice(
    config: TrainNICEConfig,
):
    torch.set_float32_matmul_precision("high")

    device = torch.device(config.device)
    dtype = get_dtype(config.dtype)

    model = NICEModel(
        config=config.nice_config,
        dataset_config=config.dataset_config,
        compile=config.compile_model,
    )
    model.to(device=device, dtype=dtype)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        filename="checkpoint/{epoch:02d}-{val/loss:.2f}",
        save_top_k=config.save_top_k,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(
        save_dir=config.log_path,
        name=config.nice_config.model_name,
        default_hp_metric=False,
    )

    trainer = Trainer(
        max_epochs=config.nice_config.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
        accelerator=config.device,
        precision=config.precision,
    )

    trainer.fit(model, ckpt_path=config.checkpoint_path)
    return model


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
