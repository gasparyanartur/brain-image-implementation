from collections.abc import Iterable
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
from transformers import CLIPVisionModelWithProjection
import tqdm
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_image_variation import (
    StableDiffusionImageVariationPipeline,
)

from brain_image.configs import get_device
from brain_image.model.img_encoder import CLIPImageEncoder, VAEImageEncoder, model_name_to_hf_name
from brain_image.utils import DTYPE


class ReconstructionPipeline:
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        noise_scheduler: DDIMScheduler,
        clip_encoder: CLIPImageEncoder,
        vae_encoder: VAEImageEncoder,
        **kwargs: dict,
    ):
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.clip_encoder = clip_encoder
        self.vae_encoder = vae_encoder


    @classmethod
    def from_stable_diffusion(
        cls,
        model_name: str = "sd_variations_v2",
        device: torch.device = get_device(),
        dtype: torch.dtype = DTYPE,
        cond_encoder_name: str = "clip_vitl14",
        **kwargs: dict,
    ):
        base_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_name_to_hf_name(model_name),
            torch_dtype=dtype,
        ).to(device)

        unet = base_pipe.unet
        vae = base_pipe.vae
        noise_scheduler = base_pipe.scheduler
        clip_encoder = CLIPImageEncoder(cond_encoder_name)
        vae_encoder = VAEImageEncoder(model_name)

        return cls(
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            clip_encoder=clip_encoder,
            vae_encoder=vae_encoder,
            **kwargs,
        )

    def reconstruct_target(
        self,
        target: torch.Tensor,
        low_level_image: torch.Tensor | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        noise_strength: float = 1.0,
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
        device = self.vae.device
        condition_latent = self.clip_encoder.encode(target.to(device))

        if low_level_image is not None:
            low_level_latent = self.vae_encoder.encode(low_level_image.to(device))
        else:
            low_level_latent = None

        return self.reconstruct_latents(
            condition_latent,
            low_level_latent,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            noise_strength=noise_strength,
            seed=seed,
            backend=backend,
            extra_step_kwargs=extra_step_kwargs,
        )

    def reconstruct_latents(
        self,
        conditioning_latent: torch.Tensor,  # <batch_size, 768>
        low_image_latent: torch.Tensor | None = None,  # <batch_size, 4, 64, 64>
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        noise_strength: float = 1.0,
        progress_bar: bool = False,
        seed: int = 0,
        backend: Literal[
            "stable_diffusion", "versatile_diffusion"
        ] = "stable_diffusion",
        extra_step_kwargs: dict = {},
    ) -> torch.Tensor:
        if backend != "stable_diffusion":
            raise NotImplementedError(f"backend {backend} is not implemented")

        device = self.vae.device
        generator = torch.Generator(device=device).manual_seed(seed)

        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(self.vae.config["block_out_channels"]) - 1)
        sample_size = self.unet.config["sample_size"]
        channels = self.unet.config["in_channels"]
        height, width = sample_size * vae_scale_factor, sample_size * vae_scale_factor
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

        reconstruction = self.vae_encoder.decode(latents)
        return reconstruction

    def compile(self):
        self.unet = torch.compile(self.unet)
        self.vae = torch.compile(self.vae)
        self.clip_encoder = torch.compile(self.clip_encoder)
        self.vae_encoder = torch.compile(self.vae_encoder)