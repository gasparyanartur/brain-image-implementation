from abc import ABC
import logging
import os
from pathlib import Path
import tomllib
from pydantic import BaseModel
import torch


DEFAULT_BATCH_SIZE = 32


class BaseConfig(BaseModel, ABC):
    @classmethod
    def from_hydra_config(cls, cfg):
        """Create an instance of the config class from a Hydra config."""
        return cls(**cfg)


def _resolve_workspace_dir() -> Path:
    curr_file = Path(__file__)

    # Find the root of the project
    while curr_file.parent != curr_file:
        if (curr_file / "pyproject.toml").exists():
            with open(curr_file / "pyproject.toml", "rb") as f:
                project_info = tomllib.load(f)

            if (
                "name" in project_info["project"]
                and project_info["project"]["name"] == "brain_image"
            ):
                return curr_file
            else:
                logging.warning(
                    f"Found pyproject.toml but it does not contain the correct project name. Expected 'brain_image' but got '{project_info['project']['name']}'"
                )
        curr_file = curr_file.parent

    raise RuntimeError("Could not find the root of the project")


class GlobalConfig:
    WORKSPACE_DIR: Path = Path(
        os.environ.get("PROJECT_WORKSPACE_DIR", _resolve_workspace_dir())
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
