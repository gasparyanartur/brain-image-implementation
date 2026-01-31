

from abc import ABC
import logging
from pathlib import Path
from typing import Literal, cast
from torch import nn, Tensor
import torch

from brain_image.configs import BaseConfig
from brain_image.utils import find_module_content_in_state_dict

EEG_ENCODER = Literal["atms", "nice", "dummy"]

class EEGEncoderConfig(BaseConfig):
    eeg_encoder: EEG_ENCODER
    d_channels: int | None = None
    d_time: int = 250
    d_output: int = 768
    d_eeg: int = 40

class EEGEncoder(ABC, nn.Module):
    def forward(self, eeg_data: Tensor, sub: Tensor | None = None, *args, **kwargs) -> Tensor:
        raise NotImplementedError()


def resolve_eeg_encoder_config(config: EEGEncoderConfig | dict) -> EEGEncoderConfig:
    if isinstance(config, EEGEncoderConfig):
        return config 
    
    match config["eeg_encoder"]:
        case "nice":
            from brain_image.model.eeg_encoder.nice import NiceEEGEncoderConfig
            return NiceEEGEncoderConfig(**config)
        
        case "atms":
            from brain_image.model.eeg_encoder.atms import AtmsEEGEncoderConfig
            return AtmsEEGEncoderConfig(**config)
        
        case "dummy":
            from brain_image.model.eeg_encoder.dummy import DummyEEGEncoderConfig
            return DummyEEGEncoderConfig(**config)
        
        case _:
            raise ValueError(f"Unknown EEG encoder: {config['eeg_encoder']}")


def create_eeg_encoder(
    config: EEGEncoderConfig | dict, checkpoint_path: Path | None = None
) -> EEGEncoder:
    config = resolve_eeg_encoder_config(config)
    match config.eeg_encoder:
        case "nice":
            from brain_image.model.eeg_encoder.nice import NiceEEGEncoder, NiceEEGEncoderConfig
            model_name = "nice"
            encoder = NiceEEGEncoder(cast(NiceEEGEncoderConfig, config))
        case "atms":
            from brain_image.model.eeg_encoder.atms import AtmsEEGEncoder, AtmsEEGEncoderConfig
            model_name = "atms"
            encoder = AtmsEEGEncoder(cast(AtmsEEGEncoderConfig, config))
        case "dummy":
            from brain_image.model.eeg_encoder.dummy import DummyEEGEncoder, DummyEEGEncoderConfig
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

