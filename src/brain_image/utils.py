from typing import Any
import torch
import os


def get_dtype(dtype: str) -> torch.dtype:
    """Convert a string to a torch dtype."""
    match dtype:
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case "bfloat16":
            return torch.bfloat16
        case "int8":
            return torch.int8
        case "int16":
            return torch.int16
        case "int32":
            return torch.int32
        case "int64":
            return torch.int64
        case "bool":
            return torch.bool
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


if "DEVICE" in os.environ:
    DEVICE = torch.device(os.environ["DEVICE"])
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "DTYPE" in os.environ:
    DTYPE = get_dtype(os.environ["DTYPE"])
else:
    DTYPE = torch.float16


def update_config_with_nested_key(
    key: str, value: Any, config: dict[str, Any]
) -> dict[str, Any]:
    """Update a config with a nested key, creating intermediate dicts as needed."""
    config = {**config}
    if "." in key:
        nested_key, sub_key = key.split(".", 1)
        # If the nested_key does not exist or is not a dict, create it as a dict
        if nested_key not in config or not isinstance(config[nested_key], dict):
            config[nested_key] = {}
        new_config = update_config_with_nested_key(sub_key, value, config[nested_key])
        config[nested_key] = new_config
    else:
        config[key] = value
    return config
