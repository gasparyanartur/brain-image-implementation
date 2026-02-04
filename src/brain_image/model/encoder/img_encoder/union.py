import logging
from pathlib import Path

import torch
from brain_image.model.encoder.img_encoder.img_encoder import BaseImageEncoder, CLIPImageEncoder, DreamsimImageEncoder, DummyImageEncoder, VAEImageEncoder


import typing


ImageEncoderName = typing.Literal[
    "clip_vitl14",
    "clip_vith14",
    "clip_vitb32",
    "sd_variations_v2",
    "ip_sdxl_turbo",
    "ip_sdxl_turbo_256",
    "ip_sdxl_turbo_128",
    "synclr_vitb16",
    "aligned_synclr_vitb16",
    "unaligned_synclr_vitb16",
    "dummy_768",
]
VAE_ENCODER = typing.Literal[
    "sd_variations_v2", "ip_sdxl_turbo", "ip_sdxl_turbo_256", "ip_sdxl_turbo_128"
]
DREAMSIM_IMAGE_ENCODER = typing.Literal[
    "synclr_vitb16", "unaligned_synclr_vitb16", "aligned_synclr_vitb16", "dummy_768"
]

IMAGE_ENCODER_DIM: dict[ImageEncoderName, int] = {
    "clip_vitl14": 768,
    "clip_vith14": 1024,
    "clip_vitb32": 512,
    "sd_variations_v2": 768,
    "ip_sdxl_turbo": 1024,
    "ip_sdxl_turbo_256": 1024,
    "ip_sdxl_turbo_128": 1024,
    "synclr_vitb16": 768,
    "aligned_synclr_vitb16": 768,
    "unaligned_synclr_vitb16": 768,
    "dummy_768": 768
}


def load_image_encoder(
    model_name: ImageEncoderName,
    models_path: Path = Path("models"),
    download_weights: bool = True,
    compile: bool = True,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    *args,
    **kwargs,
) -> BaseImageEncoder:
    logging.info(f"Loading image encoder for model {model_name} on device {device}")

    match model_name:
        case "clip_vitl14" | "clip_vith14" | "clip_vitb32":
            model = CLIPImageEncoder(model_name, *args, **kwargs)
        case (
            "sd_variations_v2"
            | "ip_sdxl_turbo"
            | "ip_sdxl_turbo_256"
            | "ip_sdxl_turbo_128"
        ):
            if model_name == "ip_sdxl_turbo_256":
                kwargs["img_width"] = 256
            elif model_name == "ip_sdxl_turbo_128":
                kwargs["img_width"] = 128
            model = VAEImageEncoder(model_name, *args, **kwargs)
        case "synclr_vitb16" | "aligned_synclr_vitb16" | "unaligned_synclr_vitb16":
            model = DreamsimImageEncoder(
                model_name,
                *args,
                models_path=models_path,
                download_weights=download_weights,
                **kwargs,
            )
        case "dummy_768" | "dummy" | "dummy_1024":
            model = DummyImageEncoder(model_name, *args, **kwargs)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    model.requires_grad_(False)
    model.eval()

    if compile:
        logging.info("Compiling model")
        model = torch.compile(model)

    if device is not None:
        logging.info(f"Moving model to device: {device}")
        model.to(device)

    if dtype is not None:
        logging.info(f"Casting model to dtype: {dtype}")
        model = model.to(dtype=dtype)

    return typing.cast(BaseImageEncoder, model)


def load_vae_encoder(model_name: VAE_ENCODER, *args, **kwargs) -> VAEImageEncoder:
    enc = typing.cast(VAEImageEncoder, load_image_encoder(model_name, *args, **kwargs))
    return enc


