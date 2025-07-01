from abc import ABC
import os
from pathlib import Path
from pydantic import BaseModel


DEFAULT_BATCH_SIZE = 32


class BaseConfig(BaseModel, ABC):
    @classmethod
    def from_hydra_config(cls, cfg):
        """Create an instance of the config class from a Hydra config."""
        return cls(**cfg)


class GlobalConfig:
    WORKSPACE_DIR: Path = Path(
        os.environ.get("PROJECT_WORKSPACE_DIR", Path(__file__).parent.parent)
    )
    CONFIGS_DIR: Path = WORKSPACE_DIR / "src" / "configs"
    DATA_DIR: Path = WORKSPACE_DIR / "data"
