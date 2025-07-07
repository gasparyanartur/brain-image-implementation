from abc import ABC
import logging
import os
from pathlib import Path
from pydantic import BaseModel
import torch


DEFAULT_BATCH_SIZE = 32


class BaseConfig(BaseModel, ABC):
    @classmethod
    def from_hydra_config(cls, cfg):
        """Create an instance of the config class from a Hydra config."""
        return cls(**cfg)


class GlobalConfig:
    WORKSPACE_DIR: Path = Path(
        os.environ.get("PROJECT_WORKSPACE_DIR", Path(__file__).parent)
    )
    CONFIGS_DIR: Path = WORKSPACE_DIR / "src" / "brain_image" / "configs"
    DATA_DIR: Path = WORKSPACE_DIR / "data"


_device: torch.device | None = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.cuda.is_available():
            logging.info("Found CUDA device, using cuda")
            _device = torch.device("cuda")
        else:
            logging.info("No CUDA device found, using cpu")
            _device = torch.device("cpu")

    return _device
