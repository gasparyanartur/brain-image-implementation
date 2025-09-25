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


def model_name_to_hf_name(model_name: str) -> str:
    match model_name:
        case "clip_vitl14":
            return "openai/clip-vit-large-patch14"
        case "sd_variations_v2":
            return "lambdalabs/sd-image-variations-diffusers"
        case "synclr_vitb16":
            return "facebook/dino-vitb16"
        case _:
            raise ValueError(f"Unknown model name: {model_name}")


class BaseImageEncoder(nn.Module):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__()
        self.model_name = model_name

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.encode(img)


def load_image_encoder(
    model_name: str,
    models_path: Path = Path("models"),
    download_weights: bool = True,
    compile: bool = True,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    *args,
    **kwargs,
) -> BaseImageEncoder:
    match model_name:
        case "clip_vitl14":
            model = CLIPImageEncoder(*args, **kwargs)
        case "sd_variations_v2":
            model = VAEImageEncoder(*args, **kwargs)
        case "aligned_synclr_vitb16" | "unaligned_synclr_vitb16":
            model = SynCLRImageEncoder(
                *args,
                models_path=models_path,
                download_weights=download_weights,
                **kwargs,
            )
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    model.requires_grad_(False)
    model.eval()

    if compile:
        model = torch.compile(model)

    if device is not None:
        model.to(device)

    model = model.to(dtype=dtype)

    return typing.cast(BaseImageEncoder, model)


class CLIPImageEncoder(BaseImageEncoder):
    def __init__(self, model_name: str = "clip_vitl14", *args, **kwargs):
        super().__init__(model_name=model_name)

        hf_name = model_name_to_hf_name(model_name)

        self.processor = CLIPImageProcessor.from_pretrained(hf_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(hf_name)

        self.model.requires_grad_(False)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        img = self.processor(images, return_tensors="pt").pixel_values.to(
            self.model.device
        )
        return self.model(img).image_embeds


class VAEImageEncoder(BaseImageEncoder):
    def __init__(self, model_name: str = "sd_variations_v2", *args, **kwargs):
        super().__init__(model_name=model_name)
        hf_name = model_name_to_hf_name(model_name)

        self.vae = AutoencoderKL.from_pretrained(hf_name, subfolder="vae")
        self.vae_scale_factor = 2 ** (len(self.vae.config["block_out_channels"]) - 1)
        self.preprocessor = tv2.Compose(
            [
                Resize(
                    (512, 512),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                ToDtype(torch.float32, scale=True),
            ]
        )
        self.feature_extractor = CLIPImageProcessor.from_pretrained(hf_name, subfolder="feature_extractor")
        self.processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.preprocessor.requires_grad_(False)
        self.vae.requires_grad_(False)

    @torch.compiler.disable(recursive=True)
    def _to_image(self, img: torch.Tensor):
        return tv2.functional.to_image(img)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        img = self._to_image(img)   # type: ignore
        img = self.preprocessor(img)
        img = self.processor.preprocess(img)

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


class SynCLRImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        model_name: typing.Literal[
            "unaligned_synclr_vitb16", "aligned_synclr_vitb16"
        ] = "unaligned_synclr_vitb16",
        download_weights: bool = True,
        models_path: Path = Path("models"),
        *args,
        **kwargs,
    ):
        super().__init__(model_name)

        models_path_str = str(models_path)
        model_name_parts = model_name.split("_")
        if model_name_parts[0] == "unaligned":
            self.aligned = False
        elif model_name_parts[1] == "aligned":
            self.aligned = True
        else:
            raise ValueError(
                f"Invalid model name: {model_name} - Could not recognize 'aligned' variable {model_name_parts[0]}"
            )

        if model_name_parts[-1] == "vitb16":
            self.patch_size = 16
        elif model_name_parts[-1] == "vitl14":
            self.patch_size = 14
        else:
            raise ValueError(
                f"Invalid model name: {model_name} - Could not recognize 'patch_size' variable {model_name_parts[-1]}"
            )

        model_url = "_".join(model_name_parts[1:])
        if download_weights:
            dreamsim.model.download_weights(
                cache_dir=models_path_str, dreamsim_type=model_url
            )

        hf_name = model_name_to_hf_name(model_url)
        processor = ViTImageProcessor.from_pretrained(hf_name)

        if self.aligned:
            self.model = PerceptualModel(
                model_type=model_url,
                normalize_embeds=False,
                stride=self.patch_size,  # type: ignore
                load_dir=models_path_str,
                baseline=True,
            )

        else:
            self.model, _ = dreamsim.dreamsim(
                dreamsim_type=model_url,
                cache_dir=models_path_str,
                normalize_embeds=False,
            )

        self.processor = processor
        self.model.requires_grad_(False)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        img = self.processor(img, return_tensors="pt").pixel_values.to(
            self.model.device
        )
        latent = self.model.embed(img)  # type: ignore

        return latent
