from pathlib import Path
import torch
from brain_image.model.eeg_encoder.eeg_encoder import EEGEncoder
from brain_image.model.eeg_encoder.nice import NiceEEGEncoder, NiceConfig
from brain_image.model.eeg_encoder.atms import AtmsEEGEncoder
from brain_image.utils import find_module_content_in_state_dict


def create_eeg_encoder(
    name: str, config=None, checkpoint_path: Path | None = None
) -> EEGEncoder:
    match name:
        case "nice":
            encoder = NiceEEGEncoder(config if config is not None else NiceConfig())
        case "atms":
            encoder = AtmsEEGEncoder()
        case _:
            raise ValueError(f"Unknown encoder name: {name}")

    if checkpoint_path is None:
        return encoder

    checkpoint = torch.load(checkpoint_path)

    eeg_encoder_state_dict = find_module_content_in_state_dict(
        "state_dict", checkpoint, module_name="eeg_encoder"
    )
    if not eeg_encoder_state_dict:
        raise ValueError("Could not find EEG encoder in checkpoint")

    encoder.load_state_dict(eeg_encoder_state_dict)
    return encoder


__all__ = ["EEGEncoder", "NiceEEGEncoder", "AtmsEEGEncoder"]
