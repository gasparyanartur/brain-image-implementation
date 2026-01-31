from typing import Literal

from brain_image.model.encoder.eeg_encoder.union import EEGEncoderName
from brain_image.model.encoder.img_encoder import ImageEncoderName


EncoderName = EEGEncoderName | ImageEncoderName
LatentName = Literal[
    "prior_img_latent", "eeg_latent", "align_img_latent", "low_level_latent"
]