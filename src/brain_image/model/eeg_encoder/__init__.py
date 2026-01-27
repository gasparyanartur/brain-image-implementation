import logging
from pathlib import Path
from typing import cast
import torch
from brain_image.model.eeg_encoder.eeg_encoder import EEGEncoder, EEGEncoderConfig
from brain_image.model.eeg_encoder.nice import NiceEEGEncoder, NiceEEGEncoderConfig
from brain_image.model.eeg_encoder.atms import AtmsEEGEncoder, AtmsEEGEncoderConfig
from brain_image.utils import find_module_content_in_state_dict


def create_eeg_encoder(
    config: EEGEncoderConfig, checkpoint_path: Path | None = None
) -> EEGEncoder:
    match config.eeg_encoder:
        case "nice":
            model_name = "nice"
            encoder = NiceEEGEncoder(cast(NiceEEGEncoderConfig, config))
        case "atms":
            model_name = "atms"
            encoder = AtmsEEGEncoder(cast(AtmsEEGEncoderConfig, config))
        case _:
            raise ValueError(f"Unknown encoder name: {name}")

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


__all__ = ["EEGEncoder", "NiceEEGEncoder", "AtmsEEGEncoder"]
