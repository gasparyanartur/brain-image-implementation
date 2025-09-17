from collections.abc import Callable
import typing
from torch import nn
import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from torchvision.transforms import v2 as tv2
from torchvision.transforms.v2 import (
    ToImage,
    ToDtype,
    Resize,
    Normalize,
    InterpolationMode,
)


class BaseImageEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.encode(img)


def hf_name_to_model_name(hf_name: str) -> str:
    match hf_name:
        case "openai/clip-vit-large-patch14":
            return "clip_vitl14"
        case "lambdalabs/sd-image-variations-diffusers":
            return "sd_variations_v2"
        case _:
            raise ValueError(f"Unknown HF model name: {hf_name}")


class CLIPImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        hf_model_name: str = "openai/clip-vit-large-patch14",
        use_native_processor: bool = False,
    ):
        super().__init__(model_name=hf_name_to_model_name(hf_model_name))

        if use_native_processor:
            proc = CLIPImageProcessor.from_pretrained(hf_model_name)
            self.processor = lambda x: proc.preprocess(
                x, return_tensors="pt"
            ).pixel_values
        else:
            self.processor = tv2.Compose(
                [
                    ToImage(),
                    ToDtype(torch.float32, scale=True),
                    Resize(
                        (224, 224),
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=False,
                    ),
                    Normalize(
                        [0.48145466, 0.4578275, 0.40821073],
                        [0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        self.model = CLIPVisionModelWithProjection.from_pretrained(hf_model_name)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        img = self.processor(images)
        out = self.model(img).image_embeds
        return out


class VAEImageEncoder(BaseImageEncoder):
    def __init__(self, hf_model_name: str = "lambdalabs/sd-image-variations-diffusers"):
        model_name = hf_name_to_model_name(hf_model_name)

        super().__init__(model_name=model_name)

        self.processor = tv2.Compose(
            [
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Resize(
                    (512, 512),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
        self.vae = AutoencoderKL.from_pretrained(hf_model_name, subfolder="vae")
        self.vae_scale_factor = 2 ** (len(self.vae.config["block_out_channels"]) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        img = self.processor(img)
        img = self.image_processor.preprocess(img)

        latent = (
            self.vae.encode(img).latent_dist.sample()  # type: ignore
            * self.vae.config["scaling_factor"]
        )

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        img = self.vae.decode(latent / self.vae.config["scaling_factor"]).sample  # type: ignore
        img = typing.cast(
            torch.Tensor, self.image_processor.postprocess(img, output_type="pt")
        )
        return img


class SynCLRImageEncoder(BaseImageEncoder):
    def __init__(
        self, model_name: str = "facebookresearch/synclr-50", patch_size: int = 16
    ):
        super().__init__()
        # TODO: Implement SynCLR image encoder
        raise NotImplementedError
