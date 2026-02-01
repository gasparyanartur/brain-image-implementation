from __future__ import annotations

from abc import ABC
from functools import lru_cache
import logging
import os
from pathlib import Path
import tomllib
from typing import Any, Generic, Mapping, TypeVar, cast
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError
from hydra.utils import instantiate
import torch

from brain_image.utils import flatten_configs


C = TypeVar("C", bound=BaseModel)


def _instantiate_targets(obj: Any) -> Any:
    """Recursively instantiate Hydra nodes that declare a target class.

    Supports both Hydra's `_target_` and the legacy `__target__` key.
    """
    if isinstance(obj, dict):
        if "_target_" in obj or "__target__" in obj:
            payload = dict(obj)
            if "__target__" in payload and "_target_" not in payload:
                payload["_target_"] = payload.pop("__target__")
            return instantiate(OmegaConf.create(payload), _convert_="all")

        return {k: _instantiate_targets(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_instantiate_targets(v) for v in obj]

    return obj



class BaseConfig(BaseModel, ABC, Generic[C]):
    @classmethod
    def from_hydra_config(cls, cfg: DictConfig, instantiate: bool = True) -> C:
        logging.info(f"Constructing from Hydra config " + str(cfg))
        raw_dict = OmegaConf.to_container(cfg, resolve=True)

        if not isinstance(raw_dict, dict):
            raise ValueError("Config must be a dictionary")

        if not all(isinstance(k, str) for k in raw_dict):
            raise ValueError("Config keys must be strings")

        raw_dict = cast(Mapping[str, Any], raw_dict)
        
        if instantiate:
            raw_dict = _instantiate_targets(raw_dict)

        try:
            obj = cls(**raw_dict)

        except ValidationError as e:
            logging.error(f"Attempted to construct {cls.__name__} from dictionary with keys:")
            for k, v in flatten_configs(raw_dict).items():
                logging.error(f"{k}: type: {type(v)}, value: {v}")
            raise e

        return cast(C, obj)

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
    if (override_device := os.environ.get("OVERRIDE_DEVICE")) is not None:
        return override_device
        
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device(get_device_str())

    return _device
