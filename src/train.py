from pathlib import Path
from typing import Literal

from configs import BaseConfig


class TrainConfig(BaseConfig):
    config_tag: str = "train"

    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    compile_model: bool = True
    log_path: Path = Path("logs")
    checkpoint_path: Path | None = None
