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
    curr_path = Path(__file__)
    logging.info(f"Resolving workspace directory, traversing up from starting from {curr_path}")

    # Find the root of the project
    while curr_path.parent != curr_path:
        curr_path = curr_path.parent
        logging.info(f"Current directory: {curr_path}")

        if (curr_path / "pyproject.toml").exists():
            logging.info("Found pyproject.toml")
            with open(curr_path / "pyproject.toml", "rb") as f:
                project_info = tomllib.load(f)

            if (
                "name" in project_info["project"]
                and project_info["project"]["name"] == "brain_image"
            ):
                return curr_path
            else:
                logging.warning(
                    f"Found pyproject.toml but it does not contain the correct project name. Expected 'brain_image' but got '{project_info['project']['name']}'"
                )

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
