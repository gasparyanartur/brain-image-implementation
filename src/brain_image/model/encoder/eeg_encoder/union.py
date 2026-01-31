from pydantic import Field
from brain_image.model.encoder.eeg_encoder.eeg_encoder import EEGEncoder, EEGEncoderConfig
from brain_image.utils import find_module_content_in_state_dict, flatten_configs

from brain_image.model.encoder.eeg_encoder.nice import NiceEEGEncoder, NiceEEGEncoderConfig
from brain_image.model.encoder.eeg_encoder.atms import AtmsEEGEncoder, AtmsEEGEncoderConfig
from brain_image.model.encoder.eeg_encoder.dummy import DummyEEGEncoder, DummyEEGEncoderConfig

import torch


import logging
from pathlib import Path
from typing import Literal, cast


def resolve_eeg_encoder_config(config: EEGEncoderConfig | dict) -> EEGEncoderConfig:
    if isinstance(config, EEGEncoderConfig):
        return config

    match config["eeg_encoder"]:
        case "nice":
            return NiceEEGEncoderConfig(**config)

        case "atms":
            return AtmsEEGEncoderConfig(**config)

        case "dummy":
            return DummyEEGEncoderConfig(**config)

        case _:
            raise ValueError(f"Unknown EEG encoder: {config['eeg_encoder']}")


def create_eeg_encoder(
    config: EEGEncoderConfig | dict, checkpoint_path: Path | None = None
) -> EEGEncoder:
    config = resolve_eeg_encoder_config(config)

    logging.info(f"Creating EEG encoder with configs:")
    for k, v in flatten_configs(config).items():
        logging.info(f"  {k}: {v}")

    match config.eeg_encoder:
        case "nice":
            model_name = "nice"
            encoder = NiceEEGEncoder(cast(NiceEEGEncoderConfig, config))
        case "atms":
            model_name = "atms"
            encoder = AtmsEEGEncoder(cast(AtmsEEGEncoderConfig, config))
        case "dummy":
            model_name = "dummy"
            encoder = DummyEEGEncoder(cast(DummyEEGEncoderConfig, config))
        case _:
            raise ValueError(f"Unknown encoder name: {config.eeg_encoder}")

    logging.info(f"Using {model_name} EEG encoder")
    if checkpoint_path is None:
        return encoder

    logging.info(f"Loading EEG checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    eeg_encoder_state_dict = find_module_content_in_state_dict(
        "state_dict", checkpoint, module_name="eeg_encoder"
    )
    if not eeg_encoder_state_dict:
        raise ValueError("Could not find EEG encoder in checkpoint")

    encoder.load_state_dict(eeg_encoder_state_dict)
    return encoder


EEGEncoderName = Literal["atms", "nice", "dummy"]
EEGEncoderConfigType = NiceEEGEncoderConfig | AtmsEEGEncoderConfig | DummyEEGEncoderConfig
EEGEncoderField = Field(discriminator="eeg_encoder")
