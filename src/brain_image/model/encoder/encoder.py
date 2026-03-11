from typing import Literal

from brain_image.model.encoder.eeg_encoder.union import EEGEncoderName
from brain_image.model.encoder.img_encoder.union import IMAGE_ENCODER_DIM, ImageEncoderName
from brain_image.model.encoder.text_encoder.union import TEXT_ENCODER_DIM, TextEncoderName


EncoderName = EEGEncoderName | ImageEncoderName | TextEncoderName
AlignEncoderName = ImageEncoderName | TextEncoderName

ALIGN_ENCODER_DIM: dict[str, int] = {**IMAGE_ENCODER_DIM, **TEXT_ENCODER_DIM}  # type: ignore[arg-type]


def get_align_encoder_dim(name: AlignEncoderName) -> int:
    if name not in ALIGN_ENCODER_DIM:
        raise ValueError(f"Unknown align encoder name: {name!r}. Available: {list(ALIGN_ENCODER_DIM)}")
    return ALIGN_ENCODER_DIM[name]
LatentName = Literal[
    "prior_img_latent", "eeg_latent", "align_img_latent", "low_level_latent"
]