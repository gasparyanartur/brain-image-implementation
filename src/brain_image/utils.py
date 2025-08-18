from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any
import dotenv
from huggingface_hub import login
import torch
import os

import matplotlib.pyplot as plt


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    dotenv.load_dotenv()

    torch.set_float32_matmul_precision('high')

    tok = os.environ.get("HF_API_TOKEN")
    if not tok:
        raise ValueError("HF_API_TOKEN is not set. Please set it in the .env file")

    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Using dtype: {DTYPE}")
    logging.info(f"Using directory: {os.getcwd()}")
    login(token=tok)


def get_mean_gradients(model: torch.nn.Module) -> torch.Tensor | None:
    grads = [p.grad.norm(dim=-1).mean() for p in model.parameters() if p.grad is not None]
    if len(grads) == 0:
        return None
    return torch.stack(grads).mean()