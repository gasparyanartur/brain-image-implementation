import logging
from typing import Literal

import hydra
from omegaconf import DictConfig
from data import EEGDatasetConfig
from model import NICEConfig, NICEModel
from train import TrainConfig


import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


from pathlib import Path

from utils import get_dtype


class TrainNICEConfig(TrainConfig):
    config_tag: str = "train_nice"

    nice_config: NICEConfig = NICEConfig()
    dataset_config: EEGDatasetConfig = EEGDatasetConfig()
    img_encoder_name: Literal["synclr", "aligned_synclr"] = "synclr"
    save_top_k: int = 1
    compile_model: bool = True
    log_path: Path = Path("logs")
    checkpoint_path: Path | None = None
    dtype: str = "float16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_nice(
    config: TrainNICEConfig,
):
    torch.set_float32_matmul_precision("medium")

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
        name=config.img_encoder_name,
        default_hp_metric=False,
    )

    trainer = Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
        accelerator=config.device,
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
