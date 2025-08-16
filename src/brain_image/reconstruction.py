import logging
from typing import Callable, Literal, cast

import torch
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    Normalize,
    ToDtype,
    ToImage,
)
from torchvision.transforms.v2 import GaussianBlur, Compose, RandomPerspective, Resize
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
import tqdm
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_image_variation import (
    StableDiffusionImageVariationPipeline,
)

from brain_image.configs import get_device
from brain_image.utils import DTYPE


class ReconstructionPipeline:
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        noise_scheduler: DDIMScheduler,
        image_encoder: CLIPImageProcessor,
        image_processor: VaeImageProcessor | None = None,
        cond_image_preprocessor: Callable | None = None,
        low_level_image_preprocessor: Callable | None = None,
        dtype: torch.dtype = DTYPE,
        **kwargs: dict,
    ):
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.image_encoder = image_encoder
        self.conditioning_image_preprocessor = cond_image_preprocessor or Compose(
            [
                ToImage(),
                ToDtype(dtype, scale=True),
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
        self.low_level_image_preprocessor = low_level_image_preprocessor or Compose(
            [
                ToImage(),
                ToDtype(dtype, scale=True),
                Resize(
                    (512, 512),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
        self.image_processor = image_processor


    @classmethod
    def from_stable_diffusion(
        cls,
        model_id: str = "lambdalabs/sd-image-variations-diffusers",
        device: torch.device = get_device(),
        dtype: torch.dtype = DTYPE,
        revision: str = "v2.0",
        cond_image_preprocessor: Callable | None = None,
        low_level_image_preprocessor: Callable | None = None,
        **kwargs: dict,
    ):
        base_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
        ).to(device)

        unet = base_pipe.unet
        vae = base_pipe.vae
        noise_scheduler = base_pipe.scheduler
        image_encoder = base_pipe.image_encoder
        image_processor = base_pipe.image_processor

        return cls(
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
            cond_image_preprocessor=cond_image_preprocessor,
            low_level_image_preprocessor=low_level_image_preprocessor,
            dtype=dtype,
            **kwargs,
        )

    def reconstruct_target(
        self,
        target: torch.Tensor,
        low_level_image: torch.Tensor | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        noise_strength: float = 1.0,
        device: torch.device = get_device(),
        seed: int = 0,
        backend: Literal[
            "stable_diffusion", "versatile_diffusion"
        ] = "stable_diffusion",
        extra_step_kwargs: dict = {},
    ):
        """
        Reconstructs the target image by converting it to a CLIP latent space and using it as a conditioning signal.

        Args:
            image <batch_size, channels, height, width>: The image to reconstruct.
            low_level_image <batch_size, channels, height, width>: The low level image to reconstruct.
            guidance_scale: The guidance scale for the reconstruction.
            num_inference_steps: The number of inference steps for the reconstruction.
            noise_strength: The strength of the noise for the reconstruction.
            device: The device to use for the reconstruction.
            seed: The seed for the reconstruction.
            backend: The backend to use for the reconstruction.
            extra_step_kwargs: Extra step kwargs passed to the noise scheduler.

        Returns:
            reconstruction <batch_size, channels, height, width>: The reconstructed image.
        """
        condition_latent = self.encode_conditioning_image(target)
        if low_level_image is not None:
            low_level_latent = self.encode_low_level_image(low_level_image)
        else:
            low_level_latent = None

        return self.reconstruct_latents(
            condition_latent,
            low_level_latent,
            guidance_scale,
            num_inference_steps,
            noise_strength,
            device,
            seed,
            backend,
            extra_step_kwargs,
        )

    def reconstruct_latents(
        self,
        conditioning_latent: torch.Tensor,  # <batch_size, 768>
        low_image_latent: torch.Tensor | None = None,  # <batch_size, 4, 64, 64>
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        noise_strength: float = 1.0,
        device: torch.device = get_device(),
        progress_bar: bool = False,
        seed: int = 0,
        backend: Literal[
            "stable_diffusion", "versatile_diffusion"
        ] = "stable_diffusion",
        extra_step_kwargs: dict = {},
    ) -> torch.Tensor:
        if backend != "stable_diffusion":
            raise NotImplementedError(f"backend {backend} is not implemented")

        generator = torch.Generator(device=device).manual_seed(seed)

        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(self.vae.config["block_out_channels"]) - 1)
        height = self.unet.config["sample_size"] * vae_scale_factor
        width = self.unet.config["sample_size"] * vae_scale_factor
        channels = self.unet.config["in_channels"]
        batch_size = conditioning_latent.shape[0]
        latents_shape = (batch_size, channels, height // 8, width // 8)

        if backend == "stable_diffusion":
            conditioning_latent = conditioning_latent.unsqueeze(1)

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        # Prepare Latents
        if low_image_latent is not None:  # Image to image reconstruction
            if low_image_latent.shape[0] != conditioning_latent.shape[0]:
                raise ValueError(
                    "low_image_latent and conditioning_latent must have the same batch size"
                )

            low_image_latent.to(device=device, dtype=conditioning_latent.dtype)

            init_timestep = min(
                int(num_inference_steps * noise_strength), num_inference_steps
            )  # Needs at least one noise step
            t_start = max(num_inference_steps - init_timestep, 0)

            timesteps = self.noise_scheduler.timesteps[t_start:]
            latent_timestep = timesteps[:1].repeat(low_image_latent.shape[0])

            init_latents = low_image_latent

            noise = torch.randn(
                latents_shape,
                generator=generator,
                device=device,
                dtype=init_latents.dtype,
            )
            latents = self.noise_scheduler.add_noise(
                init_latents, noise, latent_timestep  # type: ignore
            )
            assert latents.shape == latents_shape, f"{latents.shape} != {latents_shape}"

        else:  # Conditioning only reconstruction
            timesteps = self.noise_scheduler.timesteps
            latents = (
                torch.randn(
                    latents_shape,
                    generator=generator,
                    device=device,
                    dtype=conditioning_latent.dtype,
                )
                * self.noise_scheduler.init_noise_sigma
            )

        if do_classifier_free_guidance:
            uncond_latent = torch.zeros_like(conditioning_latent)
            conditioning_latent = torch.cat([uncond_latent, conditioning_latent])

        # Denoising Loop
        for i, t in enumerate(tqdm.tqdm(timesteps, disable=not progress_bar, desc="Reconstructing Latents")):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t  # type: ignore
            )

            noise_pred = self.unet(latent_model_input, t, conditioning_latent).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs  # type: ignore
            ).prev_sample  # type: ignore

        # Decode Latents
        reconstruction = self.vae.decode(
            latents / self.vae.config["scaling_factor"]
        ).sample  # type: ignore

        if self.image_processor:
            reconstruction = self.image_processor.postprocess(
                reconstruction, output_type="pt"
            )
            reconstruction = cast(torch.Tensor, reconstruction)
        else:
            reconstruction = (reconstruction * 0.5 + 0.5).clamp(0, 1)

        return reconstruction

    def preprocess_conditioning_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.conditioning_image_preprocessor(image)

    def preprocess_low_level_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.low_level_image_preprocessor(image)

    def encode_conditioning_image(
        self, image: torch.Tensor
    ) -> torch.Tensor:  # Assumed not preprocessed
        """
        Encodes the conditioning image into a latent space.

        Args:
            image <batch_size, channels, height, width>: The image to encode.

        Returns:
            condition_latent <batch_size, 768>: The encoded image.
        """
        image = image.to(self.vae.device)
        processed_image = self.preprocess_conditioning_image(image)
        condition_latent = self.image_encoder(processed_image).image_embeds

        return condition_latent

    def encode_low_level_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encodes the low level image into a latent space.

        Args:
            image <batch_size, channels, height, width>: The image to encode.

        Returns:
            low_level_latent <batch_size, 4, 64, 64>: The encoded image.
        """
        image = image.to(self.vae.device)

        processed_image = self.preprocess_low_level_image(image)

        if self.image_processor:
            processed_image = self.image_processor.preprocess(processed_image)
        else:
            logging.warning(
                "Tried to encode low level image but no image processor provided, Skipping this step."
            )

        low_level_latent = (
            self.vae.encode(processed_image).latent_dist.sample()  # type: ignore
            * self.vae.config["scaling_factor"]
        )
        return low_level_latent

    def compile(self):
        self.unet = torch.compile(self.unet)
        self.vae = torch.compile(self.vae)
        self.image_encoder = torch.compile(self.image_encoder)