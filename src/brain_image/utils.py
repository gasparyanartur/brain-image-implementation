from collections.abc import Callable, Sequence
from dataclasses import dataclass
import io
import logging
from pathlib import Path
from typing import Any, Mapping
import dotenv
from huggingface_hub import login
from pydantic import BaseModel
import torch
import os
import PIL.Image
import yaml

import matplotlib.pyplot as plt


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
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=300)
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


if "DEVICE" in os.environ:
    DEVICE = torch.device(os.environ["DEVICE"])
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "DTYPE" in os.environ:
    DTYPE = get_dtype(os.environ["DTYPE"])
else:
    DTYPE = torch.float32


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


def setup():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    dotenv.load_dotenv()

    torch.set_float32_matmul_precision("high")

    tok = os.environ.get("HF_API_TOKEN")
    if not tok:
        raise ValueError("HF_API_TOKEN is not set. Please set it in the .env file")

    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Using dtype: {DTYPE}")
    logging.info(f"Using directory: {os.getcwd()}")
    login(token=tok)

def flatten_configs(configs: dict[str, Any] | BaseModel, prefix="") -> dict[str, Any]:
    flat_configs = {}

    if isinstance(configs, BaseModel):
        configs = configs.model_dump(mode="json")

    for key, value in configs.items():
        if isinstance(value, dict):
            value_dict = value
            flat_configs[prefix + key] = type(value)
            flattened_dict = flatten_configs(value_dict, prefix=f"{prefix}{key}.")
            flat_configs.update(
                flattened_dict
            )
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
    grads = [
        p.grad.norm(dim=-1).mean() for p in model.parameters() if p.grad is not None
    ]
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


def find_module_content_in_state_dict(
    key: str, state_dict: dict[str, Any], module_name: str
):
    if f"{module_name}." in key:
        return state_dict

    pure_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            result = find_module_content_in_state_dict(
                key, value, module_name=module_name
            )
            if result is not None:
                pure_dict.update(result)

        elif f"{module_name}." in key:
            pure_dict[key.replace(f"{module_name}.", "")] = value

    return pure_dict


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
