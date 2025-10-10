from abc import ABC
from functools import lru_cache
import logging
import os
from pathlib import Path
import tomllib
from pydantic import BaseModel
import torch


class BaseConfig(BaseModel, ABC):
    @classmethod
    def from_hydra_config(cls, cfg):
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


@lru_cache(maxsize=1)
def get_device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device(get_device_str())

    return _device
