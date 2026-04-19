from collections.abc import Callable, Sequence
from dataclasses import dataclass
import datetime
import hashlib
import io
import logging
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Any, Literal, cast
import uuid
import dotenv
from huggingface_hub import login
import numpy as np
from pydantic import BaseModel
import torch
import os
import PIL.Image
import yaml

from torch.nn import functional as F
from torch import nn

import re


import matplotlib.pyplot as plt


def VCLR(x: torch.Tensor) -> torch.Tensor:
    return x.detach().mean().cpu()


def casttensor(x: Any) -> torch.Tensor:
    return cast(torch.Tensor, x)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def gather_dataloader(
    loader: torch.utils.data.DataLoader,
    batch_process_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, torch.Tensor | list]:
    all_samples = {}
    tensor_keys = set()

    for batch in loader:
        if batch_process_fn is not None:
            batch = batch_process_fn(batch)

        for k, v in batch.items():
            if k not in all_samples:
                all_samples[k] = []

            if isinstance(v, torch.Tensor):
                tensor_keys.add(k)
                v = v.detach().cpu()

            all_samples[k].extend(v)

    for k in tensor_keys:
        all_samples[k] = torch.stack(all_samples[k])

    return all_samples


def current_fig_to_img():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close()

    buf.seek(0)
    img = PIL.Image.open(buf).convert("RGB")

    return img


def investigate_tensor(name: str, v: torch.Tensor) -> None:
    items = {
        "shape": tuple(v.shape),
        "dtype": v.dtype,
    }

    if v.ndim > 0:
        if v.dtype in {torch.float16, torch.float32, torch.float64}:
            items["norm"] = v.norm(dim=-1).mean().item()
            items["min"] = v.min(dim=-1).values.mean().item()
            items["max"] = v.max(dim=-1).values.mean().item()
            items["mean"] = v.mean(dim=-1).mean().item()
            items["std"] = v.std(dim=-1).mean().item()

    else:
        items["value"] = v.item()

    print(f"Name: {name}")
    for k, v in items.items():
        if isinstance(v, float):
            print(f"* {k}: {v:.3f}")
        else:
            print(f"* {k}: {v}")


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


if "DTYPE" in os.environ:
    DTYPE = get_dtype(os.environ["DTYPE"])
else:
    DTYPE = torch.float32


def update_config_with_nested_key(key: str, value: Any, config: dict[str, Any]) -> dict[str, Any]:
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


def show_image(
    *imgs,
    transforms: Sequence[str] | None = None,
    suptitle: str | None = None,
    titles: Sequence[str] | None = None,
    verbose: bool = False,
    save_path: Path | None = None,
    show_fig: bool = True,
) -> None:
    imgs = [img.detach().cpu().float() for img in imgs]
    merged_imgs = []
    for img in imgs:
        if len(img.shape) == 3:
            merged_imgs.append(img)
        elif len(img.shape) == 4:
            for i in range(img.shape[0]):
                merged_imgs.append(img[i])
        else:
            raise ValueError(f"Invalid image shape: {img.shape}")
    imgs = merged_imgs

    if transforms is None:
        transforms = ["none" for _ in imgs]

    for i in range(len(imgs)):
        imgs[i] = imgs[i].permute(1, 2, 0)

        mean_vec = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std_vec = torch.tensor([0.26862954, 0.26130258, 0.27577711])

        match transforms[i]:
            case "none":
                pass
            case "unstandardize":
                # Inverse of the normalization transform
                imgs[i] = imgs[i] * std_vec + mean_vec
            case "standardize":
                imgs[i] = (imgs[i] - mean_vec) / std_vec
            case "normalize":
                imgs[i] = imgs[i] / 255.0
            case _:
                logging.warning(f"Invalid transform: {transforms[i]}. Ignoring it.")

    if verbose:
        print("Shapes", ", ".join([str(tuple(img.shape)) for img in imgs]))
        print("Max", ", ".join([str(round(img.max().item(), 3)) for img in imgs]))
        print("Min", ", ".join([str(round(img.min().item(), 3)) for img in imgs]))

    fig, axs = plt.subplots(1, len(imgs), figsize=(10 * len(imgs), 10))
    if len(imgs) == 1:
        axs = [axs]

    if titles is not None:
        for i, title in enumerate(titles):
            axs[i].set_title(title)

    for i, img in enumerate(imgs):
        img = img.detach().cpu().float()

        axs[i].imshow(img)
        axs[i].axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if show_fig:
        plt.show()
    else:
        plt.close()


def setup_huggingface():
    tok = os.environ.get("HF_API_TOKEN")
    if not tok:
        raise ValueError("HF_API_TOKEN is not set. Please set it in the .env file")

    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

    logging.info("Configured Hugging Face Hub auth via HF_TOKEN env var")


def setup():
    from brain_image.configs import get_device_str

    dotenv.load_dotenv()

    setup_logging()
    setup_huggingface()

    torch.set_float32_matmul_precision("high")

    device_str = get_device_str()
    logging.info(f"Using device: {device_str}")
    logging.info(f"Using directory: {os.getcwd()}")


def flatten_configs(configs: Mapping[str, Any] | BaseModel, prefix="") -> dict[str, Any]:
    flat_configs = {}

    if isinstance(configs, BaseModel):
        configs = configs.model_dump(mode="json")

    for key, value in configs.items():
        if isinstance(value, Mapping):
            value_dict = value
            flat_configs[prefix + key] = type(value)
            flattened_dict = flatten_configs(value_dict, prefix=f"{prefix}{key}.")
            flat_configs.update(flattened_dict)
        else:
            flat_configs[prefix + key] = value

    return flat_configs


def init_wandb():
    try:
        import wandb

        if "WANDB_API_KEY" in os.environ:
            logging.info("WANDB_API_KEY found, attempting to login to wandb...")
            api_key = os.environ["WANDB_API_KEY"]
            logging.info("Successfully logged in to wandb")
        elif (config_path := Path("src/brain_image/configs/wandb/wandb.yaml")).exists:
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                if "api_key" in config:
                    logging.info("WANDB_API_KEY found in config, attempting to login to wandb...")
                    api_key = config["api_key"]
        else:
            logging.warning("WANDB_API_KEY not found in environment and config not found.")

        wandb.login(key=api_key)
        logging.info("Successfully logged in to wandb")
    except ImportError:
        logging.warning("wandb not available")
    except BaseException as e:
        logging.warning(f"Failed to login to wandb: {e}")


def get_mean_gradients(model: torch.nn.Module) -> torch.Tensor | None:
    grads = [p.grad.norm(dim=-1).mean() for p in model.parameters() if p.grad is not None]
    if len(grads) == 0:
        return None
    return torch.stack(grads).mean()


def state_dict_equal(state_dict1, state_dict2):
    if len(state_dict1) != len(state_dict2):
        return False
    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False
    return True


def get_subgroup_name(full_name: str, module_pattern: re.Pattern, only_subgroup: bool = False) -> str | None:
    match = module_pattern.match(full_name)
    if match is None:
        return None

    if len(match.groups()) > 1:
        raise NotImplementedError(
            f"Got multiple matches for the pattern {str(module_pattern)} in name {full_name}. Currently only one match is supported."
        )

    span = match.span(0)

    if only_subgroup:
        start_idx = span[1]

    else:
        start_idx = span[0]

    return full_name[start_idx:]


def get_submodules_with_pattern(state_dict: dict, pattern: re.Pattern | str, only_subgroup: bool = True, get_full_name: bool = False) -> dict:
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    return {
        (k if get_full_name else name): v
        for k, v in state_dict.items()
        if (name := get_subgroup_name(k, pattern, only_subgroup=only_subgroup)) is not None
    }


def key_in_dict(key: str, d: Mapping[str, Any]) -> bool:
    return key in d and d[key] is not None


@dataclass(slots=True, frozen=True)
class NormDirLen:
    norm: torch.Tensor
    dir: torch.Tensor
    len: torch.Tensor


def get_norm_dir_len(vec: torch.Tensor, eps: float = 1e-8) -> NormDirLen:
    norm = vec.norm(dim=-1, keepdim=True).detach()
    return NormDirLen(norm, vec / (norm + eps), norm.mean())


def random_word(word_len: int, seed: int | None = None) -> str:
    rng = np.random.default_rng(seed)
    min_value = ord("a")
    max_value = ord("z")
    ids = rng.integers(min_value, max_value + 1, size=word_len)
    chars = [chr(i) for i in ids]
    return "".join(chars)


def create_model_id(seed: int | None = None) -> str:
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    unique_word = random_word(6, seed=seed)
    return timestamp + unique_word


@torch.compile()
def z_scale(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.to(x.device)) / std.to(x.device)


@torch.compile()
def reverse_z_scale(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std.to(x.device) + mean.to(x.device)


@torch.compile()
def l2_scale(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1) / (x.size(-1) ** 0.5)


@torch.compile()
def reverse_l2_scale(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


@torch.compile()
def tensor_split(x: torch.Tensor, dim: int, idxs: tuple[int, ...]) -> list[torch.Tensor]:
    all_tensors = []

    curr_idx = 0
    for idx in idxs:
        part = x[..., curr_idx : curr_idx + idx]
        all_tensors.append(part)
        curr_idx += part.size(dim)

    if x.size(dim) != curr_idx:
        raise ValueError(f"Tried to split tensor of size {x.size(dim)} into {len(idxs)} parts - Received output with size {curr_idx}")

    return all_tensors


@torch.no_grad()
def find_duplicates(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError("Input tensor must be 1-dimensional")

    duplicate_idxs = []
    unique_values = set()

    for i, v in enumerate(x):
        s = v.item()
        if s in unique_values:
            duplicate_idxs.append(i)
        else:
            unique_values.add(s)

    dups = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    dups[duplicate_idxs] = 1

    return dups


def batchify_operation(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, batch_size: int) -> torch.Tensor:
    n = len(x)
    res = []
    for i in range(0, n, batch_size):
        res.append(f(x[i : i + batch_size]))
    return torch.cat(res, dim=0)


def gather_records(records: list[dict], tensor_gather: Literal["stack", "cat"] = "stack") -> dict[str, torch.Tensor]:
    if not records:
        return {}

    def gather_row(key: str, row: list[dict[str, Any]]) -> Any:
        example = row[0][key]

        if isinstance(example, torch.Tensor):
            if tensor_gather == "stack":
                return torch.stack([r[key] for r in row])
            elif tensor_gather == "cat":
                return torch.cat([r[key] for r in row])
        elif isinstance(example, list):
            return [r[key] for r in row]
        else:
            raise ValueError(f"Unsupported type {type(example)} for key {key}")

    return {k: gather_row(k, records) for k in records[0].keys()}


def prep_batch_for_logs(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    output = {}

    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                v = torch.stack(v).detach().cpu()
            else:
                try:
                    v = torch.tensor(v)
                except BaseException as e:
                    ...

        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()

        if v.dim() == 0:
            v = v.item()

        output[k] = v

    return output


def get_device_from_module(mod: nn.Module):
    return next(mod.parameters()).device


def get_dtype_from_module(mod: nn.Module):
    return next(mod.parameters()).dtype


@torch.compile()
def z_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean(dim=dim, keepdim=True)) / (x.std(dim=dim, keepdim=True) + eps)



def get_model_parameter_count(model: nn.Module) -> dict[str, int]:
    counts = {}
    for model_name, submodel in model.named_modules():
        counts[model_name] = sum(p.numel() for p in submodel.parameters())
    return counts