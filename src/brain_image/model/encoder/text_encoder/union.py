from __future__ import annotations

import logging
import typing
from pathlib import Path

import torch

from brain_image.model.encoder.text_encoder.text_encoder import (
    BaseTextEncoder,
    CLIPTextEncoder,
    GemmaTextEncoder,
    T5TextEncoder,
)


TextEncoderName = typing.Literal[
    "t5_base",
    "clip_vitl14_text",
    "gemma_embedding_300m",
]

TextEncoderChoices: list[str] = list(typing.get_args(TextEncoderName))

TEXT_ENCODER_DIM: dict[TextEncoderName, int] = {
    "t5_base": 768,
    "clip_vitl14_text": 768,
    "gemma_embedding_300m": 768,
}

_HF_NAME: dict[TextEncoderName, str] = {
    "t5_base": "t5-base",
    "clip_vitl14_text": "openai/clip-vit-large-patch14",
    "gemma_embedding_300m": "google/embeddinggemma-300m",
}


def text_encoder_name_to_hf_name(model_name: TextEncoderName) -> str:
    if model_name not in _HF_NAME:
        raise ValueError(f"Unknown text encoder model name: {model_name}")
    return _HF_NAME[model_name]


def load_text_encoder(
    model_name: TextEncoderName,
    compile: bool = False,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    *args,
    **kwargs,
) -> BaseTextEncoder:
    logging.info(f"Loading text encoder '{model_name}' on device {device}")

    hf_name = text_encoder_name_to_hf_name(model_name)

    match model_name:
        case "t5_base":
            model: BaseTextEncoder = T5TextEncoder(hf_name, *args, **kwargs)
        case "clip_vitl14_text":
            model = CLIPTextEncoder(hf_name, *args, **kwargs)
        case "gemma_embedding_300m":
            model = GemmaTextEncoder(hf_name, *args, **kwargs)
        case _:
            raise ValueError(f"Unknown text encoder model name: {model_name}")

    # Always store the logical name so TensorCache keys are stable
    model.model_name = model_name

    model.requires_grad_(False)
    model.eval()

    if compile:
        logging.info("Compiling text encoder model")
        model = torch.compile(model)  # type: ignore[assignment]

    if device is not None:
        logging.info(f"Moving text encoder to device: {device}")
        model.to(device)

    if dtype is not None:
        logging.info(f"Casting text encoder to dtype: {dtype}")
        model = model.to(dtype=dtype)

    return typing.cast(BaseTextEncoder, model)
