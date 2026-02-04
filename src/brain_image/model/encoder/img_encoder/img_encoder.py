from __future__ import annotations


from collections.abc import Callable
from pathlib import Path
import typing
from torch import nn
import torch
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    ViTImageProcessor,
)
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
import dreamsim
from dreamsim.model import PerceptualModel

from brain_image.configs import get_device_str


def model_name_to_hf_name(model_name: str) -> str:
    match model_name:
        case "clip_vitl14":
            return "openai/clip-vit-large-patch14"
        case "clip_vith14":
            return "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        case "clip_vitb32":
            return "openai/clip-vit-base-patch32"
        case "sd_variations_v2":
            return "lambdalabs/sd-image-variations-diffusers"
        case "ip_sdxl_turbo" | "ip_sdxl_turbo_256" | "ip_sdxl_turbo_128":
            return "stabilityai/sdxl-turbo"
        case "synclr_vitb16" | "aligned_synclr_vitb16" | "unaligned_synclr_vitb16":
            return "facebook/dino-vitb16"
        case _:
            raise ValueError(f"Unknown model name: {model_name}")


class BaseImageEncoder(nn.Module):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__()
        self.model_name = model_name

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.encode(img)


class CLIPImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        model_name: str = "clip_vitl14",
        preprocessor_kwargs={},
        *args,
        **kwargs,
    ):
        super().__init__(model_name=model_name)

        hf_name = model_name_to_hf_name(model_name)

        self.processor = CLIPImageProcessor.from_pretrained(hf_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(hf_name)
        self.preprocessor_kwargs = preprocessor_kwargs

        self.model.requires_grad_(False)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        extra_kwargs = {}
        if images.max() <= 3:
            extra_kwargs["do_rescale"] = False

        extra_kwargs.update(self.preprocessor_kwargs)

        return self.processor(
            images, return_tensors="pt", **extra_kwargs
        ).pixel_values.to(self.model.device)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        img = self.preprocess(images)
        return self.model(img).image_embeds


class VAEImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        model_name: str = "ip_sdxl_turbo",
        img_width: int = 512,
        img_height: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(model_name=model_name)
        hf_name = model_name_to_hf_name(model_name)

        img_height = img_height or img_width

        self.vae = AutoencoderKL.from_pretrained(hf_name, subfolder="vae")
        self.vae_scale_factor = 2 ** (len(self.vae.config["block_out_channels"]) - 1)
        self.preprocessor = tv2.Compose(
            [
                Resize(
                    (img_height, img_width),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                ToDtype(torch.float32, scale=True),
            ]
        )
        self.processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.preprocessor.requires_grad_(False)
        self.vae.requires_grad_(False)

    @torch.compiler.disable(recursive=True)
    def _to_image(self, img: torch.Tensor):
        return tv2.functional.to_image(img)

    def preprocess(
        self, img: torch.Tensor, skip_processor: bool = False
    ) -> torch.Tensor:
        img = self._to_image(img)  # type: ignore
        img = self.preprocessor(img)
        if not skip_processor:
            img = self.processor.preprocess(img)
        return img

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        img = self.preprocess(img)

        latent = (
            self.vae.encode(img).latent_dist.sample()  # type: ignore
            * self.vae.config["scaling_factor"]
        )

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        img = self.vae.decode(latent / self.vae.config["scaling_factor"]).sample  # type: ignore
        img = typing.cast(
            torch.Tensor, self.processor.postprocess(img, output_type="pt")
        )
        return img


class DreamsimImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        model_name: str = "unaligned_synclr_vitb16",
        download_weights: bool = True,
        models_path: Path = Path("models"),
        disable_grad: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(model_name)

        models_path_str = str(models_path)
        model_name_parts = model_name.split("_")
        self.preprocessor_kwargs = {}

        if len(model_name_parts) == 2:
            model_name_parts = ["unaligned"] + model_name_parts

        aligned, _, patch_size = model_name_parts

        match aligned:
            case "unaligned":
                self.aligned = False
            case "aligned":
                self.aligned = True
            case _:
                raise ValueError(
                    f"Invalid model name: {model_name} - Could not recognize 'aligned' variable {aligned}"
                )

        match patch_size:
            case "vitb16":
                self.patch_size = 16
            case "vitl14":
                self.patch_size = 14
            case _:
                raise ValueError(
                    f"Invalid model name: {model_name} - Could not recognize 'patch_size' variable {model_name_parts[-1]}"
                )

        model_url = "_".join(model_name_parts[1:])
        if download_weights:
            dreamsim.model.download_weights(
                cache_dir=models_path_str, dreamsim_type=model_url
            )

        hf_name = model_name_to_hf_name(model_name)
        processor = ViTImageProcessor.from_pretrained(hf_name)

        if not self.aligned:
            self.model = PerceptualModel(
                model_type=model_url,
                normalize_embeds=False,
                stride=str(self.patch_size),  # type: ignore
                load_dir=models_path_str,
                baseline=True,
                device=get_device_str(),
            )

        else:
            self.model, _ = dreamsim.dreamsim(
                dreamsim_type=model_url,
                cache_dir=models_path_str,
                normalize_embeds=False,
                device=get_device_str(),
            )

        self.processor = processor
        self.model.requires_grad_(not disable_grad)

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        extra_kwargs = {}
        if img.max() <= 3:
            extra_kwargs["do_rescale"] = False

        extra_kwargs.update(self.preprocessor_kwargs)
        img = self.processor(img, return_tensors="pt", **extra_kwargs).pixel_values.to(
            self.model.device
        )
        return img

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        img = self.preprocess(img)
        latent = self.model.embed(img)  # type: ignore

        return latent


class DummyImageEncoder(BaseImageEncoder):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)

        self.dim = int(model_name.split("_")[-1])

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(img)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        return torch.randn(img.shape[0], self.dim, device=img.device, dtype=torch.float)
