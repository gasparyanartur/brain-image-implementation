from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import torch
from torch.nn import functional as F

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_image_variation import (
    StableDiffusionImageVariationPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline, 
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    retrieve_timesteps, rescale_noise_cfg,  XLA_AVAILABLE
)


from brain_image.configs import get_device
from brain_image.model.encoder.img_encoder.img_encoder import (
    CLIPImageEncoder,
    VAEImageEncoder,
    model_name_to_hf_name,
)
from brain_image.model.encoder.img_encoder.union import ImageEncoderName
from brain_image.model.prior import BaseDiffusionPrior


class ReconstructionPipeline(ABC):
    @classmethod
    @abstractmethod
    def load_pretrained(cls, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def reconstruct_latents(
        self,
        conditioning_latent: torch.Tensor,  # <batch_size, 768>
        low_image_latent: torch.Tensor | None = None,  # <batch_size, 4, 64, 64>
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        noise_strength: float = 1.0,
        progress_bar: bool = False,
        seed: int = 0,
        extra_step_kwargs: dict = {},
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


from diffusers.image_processor import PipelineImageInput

class IPAdapterReconstructionPipeline(ReconstructionPipeline):
    @staticmethod
    @torch.no_grad()
    def generate_ip_adapter_embeds(
        pipe: StableDiffusionXLPipeline,
        prompt: Union[str, List[str]]  | None= None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] | None= None,
        width: Optional[int] | None = None,
        num_inference_steps: int = 50,
        timesteps: List[int] | None= None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        # Copied from Guided EEG diffusion paper
        callback_steps = None
        callback = None

        # 0. Default height and width to unet
        height = height or pipe.default_sample_size * pipe.vae_scale_factor
        width = width or pipe.default_sample_size * pipe.vae_scale_factor

        if original_size is None:
            assert height is not None and width is not None, "Must provide either original_size or height and width"
            original_size = (height, width)

        if target_size is None:
            assert height is not None and width is not None, "Must provide either target_size or height and width"
            target_size = (height, width)   

        num_images_per_prompt = num_images_per_prompt or 1

        # 1. Check inputs. Raise error if not correct
        pipe.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        pipe._guidance_scale = guidance_scale
        pipe._guidance_rescale = guidance_rescale
        pipe._clip_skip = clip_skip
        pipe._cross_attention_kwargs = cross_attention_kwargs
        pipe._denoising_end = denoising_end

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            assert prompt_embeds is not None
            batch_size = prompt_embeds.shape[0] 

        device = pipe._execution_device

        # 3. Encode input prompt
        lora_scale = (
            pipe.cross_attention_kwargs.get("scale", None) if pipe.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,  # type: ignore
            prompt_2=prompt_2,  # type: ignore
            device=device,  # type: ignore
            num_images_per_prompt=num_images_per_prompt, # type: ignore
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            negative_prompt=negative_prompt,    # type: ignore
            negative_prompt_2=negative_prompt_2,    # type: ignore
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=pipe.clip_skip,
        )
        assert num_images_per_prompt is not None

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, timesteps) # type: ignore

        # 5. Prepare latent variables
        num_channels_latents = pipe.unet.config.in_channels
        assert prompt_embeds is not None
        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt, 
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,     
            device,
            generator,
            latents,
        )
        assert latents is not None

        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1]) #type: ignore
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

        add_time_ids = pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype, # type: ignore
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = pipe._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,  # type: ignore
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if pipe.do_classifier_free_guidance:
            assert negative_prompt_embeds is not None
            assert prompt_embeds is not None
            assert negative_pooled_prompt_embeds is not None
            assert pooled_prompt_embeds is not None

            prompt_embeds = cast(torch.FloatTensor, torch.cat([negative_prompt_embeds, prompt_embeds], dim=0))
            pooled_prompt_embeds = cast(torch.FloatTensor, torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0))
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        assert prompt_embeds is not None
        assert pooled_prompt_embeds is not None

        prompt_embeds = cast(torch.FloatTensor, prompt_embeds.to(device) )
        pooled_prompt_embeds = cast(torch.FloatTensor, pooled_prompt_embeds.to(device))

        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1) 

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = pipe.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if pipe.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)
        
        if ip_adapter_embeds is not None:
            if isinstance(ip_adapter_embeds, list):  
                image_embeds = [emb.to(device=device, dtype=prompt_embeds.dtype).unsqueeze(0) for emb in ip_adapter_embeds]

            else:
                image_embeds = ip_adapter_embeds.to(device=device, dtype=prompt_embeds.dtype) 
            if pipe.do_classifier_free_guidance:
                image_embeds = cast(torch.FloatTensor, image_embeds)
                negative_image_embeds = torch.zeros_like(image_embeds)
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        timesteps = cast(list, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0) 

        # 8.1 Apply denoising_end
        if (
            pipe.denoising_end is not None
            and isinstance(pipe.denoising_end, float)
            and pipe.denoising_end > 0
            and pipe.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    pipe.scheduler.config.num_train_timesteps
                    - (pipe.denoising_end * pipe.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))   
            timesteps = timesteps[:num_inference_steps]

        assert num_inference_steps is not None

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt) 
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)     

        pipe._num_timesteps = len(timesteps)     # type: ignore
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:  # type: ignore
            for i, t in enumerate(timesteps): # type: ignore
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents # type: ignore

                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=pipe.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if pipe.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if pipe.do_classifier_free_guidance and pipe.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)     # type: ignore

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("add_text_embeds", pooled_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0): # type: ignore
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0: # type: ignore
                        step_idx = i // getattr(pipe.scheduler, "order", 1)
                        callback(step_idx, t, latents) # type: ignore

                if XLA_AVAILABLE: 
                    xm.mark_step() # type: ignore

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast

            if needs_upcasting:
                pipe.upcast_vae()
                latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype) # type: ignore

            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                pipe.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if pipe.watermark is not None:
                image = pipe.watermark.apply_watermark(image) # type: ignore

            image = pipe.image_processor.postprocess(image, output_type=output_type) # type: ignore

        # Offload all models
        pipe.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image) # type: ignore

    def __init__(self, pipe: StableDiffusionXLPipeline, **kwargs):
        self.pipe = pipe
        self.pipe.set_progress_bar_config(disable=True)

    @classmethod
    def load_pretrained(
        cls,
        model_name: str = "stabilityai/sdxl-turbo",
        ip_adapter_model_name: str = "h94/IP-Adapter",
        device: torch.device = get_device(),
        dtype=torch.float16,
        *args,
        **kwargs,
    ):

        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            device=device,
        )
        pipe.load_ip_adapter(
            ip_adapter_model_name,
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.safetensors",
            torch_dtype=dtype,
        )
        pipe.set_ip_adapter_scale(1)
        pipe.to(device)

        return cls(pipe=pipe, *args, **kwargs)

    def reconstruct_latents(
        self,
        conditioning_latent: torch.Tensor,  # <batch_size, 768>
        low_image_latent: torch.Tensor | None = None,  # <batch_size, 4, 64, 64>
        guidance_scale: float = 0,
        num_inference_steps: int = 4,
        noise_strength: float = 1.0,
        progress_bar: bool = False,
        seed: int = 0,
        generator: torch.Generator | None = None,
        extra_step_kwargs: dict = {},
        **kwargs,
    ) -> torch.Tensor:
        # As of today (2025-Oct-09), IP adapter is bugged
        # If we pass a batch of image embeddings, it will combine them across all images in the batch, rather than using each one as conditioning for each separate prompt
        # See: https://github.com/huggingface/diffusers/discussions/7933
        # As a workaround, we generate the images one-by-one
        outputs = []
        for i in tqdm.tqdm(range(conditioning_latent.size(0)), disable=not progress_bar, desc="Reconstructing latents"):
            cond_latent = [conditioning_latent[i].unsqueeze(0)]
            base_latent = cast(torch.FloatTensor, low_image_latent[i]) if low_image_latent is not None else None # TODO
            recon = self.generate_ip_adapter_embeds(
                self.pipe,
                prompt=kwargs.get("prompt", ""),
                ip_adapter_embeds=cond_latent, # type: ignore
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
                latents=base_latent
            ).images    # type: ignore
            outputs.append(recon)
            reconstructions = torch.cat(outputs)

    
        return reconstructions



class ImageVariationReconstructionPipeline(ReconstructionPipeline):
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
    def load_pretrained(
        cls,
        model_name: ImageEncoderName = "sd_variations_v2",
        device: torch.device = get_device(),
        dtype: torch.dtype = torch.float16,
        cond_encoder_name: ImageEncoderName = "clip_vitl14",
        **kwargs: dict,
    ):
        base_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_name_to_hf_name(model_name),
            torch_dtype=dtype,
        ).to(device)

        unet = base_pipe.unet
        vae = base_pipe.vae
        noise_scheduler = base_pipe.scheduler
        clip_encoder = CLIPImageEncoder(cond_encoder_name).to(
            dtype=dtype, device=device
        )
        vae_encoder = VAEImageEncoder(model_name).to(dtype=dtype, device=device)

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
        for i, t in enumerate(
            tqdm.tqdm(
                timesteps, disable=not progress_bar, desc="Reconstructing Latents"
            )
        ):
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




@torch.no_grad()
def get_reconstructions(
    conditioning: torch.Tensor,
    pipe_kwargs: dict = {},
) -> torch.Tensor:
    pipe = IPAdapterReconstructionPipeline.load_pretrained(device=conditioning.device)
    conditioning = F.normalize(conditioning, dim=-1)
    reconstruction = pipe.reconstruct_latents(conditioning, **pipe_kwargs)
    del pipe

    return reconstruction


@torch.no_grad()
def get_batched_reconstructions_from_eeg(
    prior: BaseDiffusionPrior,
    eeg_latent_normed: torch.Tensor,
    batch_size: int,
    seed: int | None = None,
    progress_bar: bool = True
):
    device = eeg_latent_normed.device 
    generator = torch.Generator(device).manual_seed(seed) if seed else None
    pred = prior.batch_generate(
        eeg_latent_normed.to(device),
        generator=generator,
        batch_size=batch_size,
    )

    reconstructions = get_reconstructions(
        pred.to(device),
        pipe_kwargs={"progress_bar": progress_bar, "generator": generator},
    )

    return reconstructions