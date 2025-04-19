from pathlib import Path
from typing import Literal
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from configs import BaseConfig
from data import EEGDatasetConfig
from model import NICEConfig, NICEModel
from utils import DEVICE


class TrainConfig(BaseConfig):
    config_tag: str = "train"

    max_epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    model_name: Literal["synclr", "aligned_synclr"] = "synclr"
    compile_model: bool = True
    log_path: Path = Path("logs")
    checkpoint_path: Path | None = None


def setup_and_train(
    config: NICEConfig = NICEConfig(),
    dataset_config: EEGDatasetConfig = EEGDatasetConfig(),
    save_top_k: int = 1,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float16,
    compile_model: bool = True,
    log_path: Path = Path("logs"),
    checkpoint_path: Path | None = None,
):
    torch.set_float32_matmul_precision("medium")

    model = NICEModel(
        config=config,
        dataset_config=dataset_config,
        compile=compile_model,
    )
    model.to(device=device, dtype=dtype)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        filename="checkpoint/{epoch:02d}-{val/loss:.2f}",
        save_top_k=save_top_k,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(
        save_dir=log_path,
        name=config.model_name,
        default_hp_metric=False,
    )

    trainer = Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        enable_progress_bar=True,
        accelerator=device,
    )

    trainer.fit(model, checkpoint_path=checkpoint_path)
    return model
