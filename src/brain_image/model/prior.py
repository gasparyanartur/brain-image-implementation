import json
import logging
from pathlib import Path
import random

from typing import Literal, cast
from PIL.Image import Image

from regex import P
import requests
import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


import logging

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from torch.nn import functional as F

from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig
from dalle2_pytorch.dalle2_pytorch import CausalTransformer, SinusoidalPosEmb, MLP

from brain_image.configs import BaseConfig


class BrainDiffusionPriorConfig(BaseConfig):
    dim: int = 768
    image_embed_dim: int = 768
    depth: int = 3
    dim_head: int = 64
    attn_dropout: float = 0.5
    ff_dropout: float = 0.5
    cond_drop_prob: float = 0.1
    image_cond_drop_prob: float = 0.0
    num_timesteps: int = 1000
    num_time_embeds: int = 1
    num_image_embeds: int = 1
    num_text_embeds: int = 1
    max_text_len: int = 0
    self_cond: bool = False
    num_output_tokens: int = 1
    rotary_emb: bool = True
    normformer: bool = True
    norm_out: bool = True
    loss_type: Literal["l2", "l1", "huber"] = "l2"
    condition_on_text_encodings: bool = False
    image_size: int = 224
    predict_x_start: bool = True
    sample_timesteps: int | None = None
    beta_schedule: Literal["cosine", "linear", "quadratic", "sigmoid"] = "cosine"
    image_embed_scale: float | None = None
    init_image_embed_l2norm: bool = False


class DiffusionPriorNetwork(nn.Module):
    # Adapted from https://github.com/lucidrains/DALLE2-pytorch/blob/main/dalle2_pytorch/dalle2_pytorch.py

    def __init__(
        self,
        dim,
        num_timesteps: int | None = 1000,
        num_time_embeds: int = 1,
        num_image_embeds: int = 1,
        num_text_embeds: int = 1,
        max_text_len: int = 256,
        self_cond: bool = False,
        depth: int = 6,
        num_output_tokens: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        self.num_time_embeds = num_time_embeds
        self.num_image_embeds = num_image_embeds
        self.num_text_embeds = num_text_embeds
        self.num_timesteps = num_timesteps

        self.to_text_embeds = nn.Sequential(
            (
                nn.Linear(dim, dim * num_text_embeds)
                if num_text_embeds > 1
                else nn.Identity()
            ),
            Rearrange("b (n d) -> b n d", n=num_text_embeds),
        )

        self.continuous_embedded_time = num_timesteps is None

        self.to_time_embeds = nn.Sequential(
            (
                nn.Embedding(num_timesteps, dim * num_time_embeds)
                if num_timesteps is not None
                else nn.Sequential(
                    SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)
                )
            ),  # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange("b (n d) -> b n d", n=num_time_embeds),
        )

        self.to_image_embeds = nn.Sequential(
            (
                nn.Linear(dim, dim * num_image_embeds)
                if num_image_embeds > 1
                else nn.Identity()
            ),
            Rearrange("b (n d) -> b n d", n=num_image_embeds),
        )

        self.learned_query = nn.Parameter(torch.randn(dim))
        self.causal_transformer = CausalTransformer(dim=dim, depth=depth, **kwargs)

        # dalle1 learned padding strategy

        self.max_text_len = max_text_len

        self.null_text_encodings = nn.Parameter(torch.randn(1, max_text_len, dim))
        self.null_text_embeds = nn.Parameter(torch.randn(1, num_text_embeds, dim))
        self.null_image_embed = nn.Parameter(torch.randn(1, dim))

        # whether to use self conditioning, Hinton's group's new ddpm technique

        self.self_cond = self_cond

        # Number of output tokens
        self.num_output_tokens = num_output_tokens

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(
            *args, text_cond_drop_prob=1.0, image_cond_drop_prob=1, **kwargs
        )
        return null_logits + (logits - null_logits) * cond_scale

    def _cond_drop_prob(
        self, prob: float, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if prob == 0.0:
            p = torch.ones((batch,), device=device, dtype=dtype)
        elif prob == 1.0:
            p = torch.zeros((batch,), device=device, dtype=dtype)
        else:
            p = torch.rand((batch,), device=device, dtype=dtype) < 1 - prob
        return p.bool()

    def forward(
        self,
        image_embed: torch.Tensor,
        diffusion_timesteps: torch.Tensor,
        *,
        text_embed: torch.Tensor | None = None,
        text_encodings: torch.Tensor | None = None,
        self_cond: torch.Tensor | None = None,
        text_cond_drop_prob: float = 0.0,
        image_cond_drop_prob=0.0,
    ):
        if text_encodings is not None:
            raise NotImplementedError(
                "text_encodings are not supported in the prior network"
            )

        if text_embed is None:
            raise NotImplementedError("text_embed is required in the prior network")

        batch, dim, device, dtype = (
            *image_embed.shape,
            image_embed.device,
            image_embed.dtype,
        )

        # setup self conditioning

        if self.self_cond:
            self_cond = self_cond or torch.zeros(
                batch, self.dim, device=device, dtype=dtype
            )
            self_cond = rearrange(self_cond, "b d -> b 1 d")

        # in section 2.2, last paragraph
        # "... consisting of encoded text, CLIP text embedding, diffusion timestep embedding, noised CLIP image embedding, final embedding for prediction"

        text_embed = self.to_text_embeds(text_embed)
        image_embed = self.to_image_embeds(image_embed)

        # classifier free guidance masks
        text_keep_mask = self._cond_drop_prob(text_cond_drop_prob, batch, device, dtype)
        text_keep_mask = rearrange(text_keep_mask, "b -> b 1 1")

        image_keep_mask = self._cond_drop_prob(
            image_cond_drop_prob, batch, device, dtype
        )
        image_keep_mask = rearrange(image_keep_mask, "b -> b 1 1")

        # make text encodings optional
        # although the paper seems to suggest it is present <--

        # mask out text embeddings with null text embeddings

        null_text_embeds = self.null_text_embeds.to(text_embed.dtype)       # type: ignore

        text_embed = torch.where(text_keep_mask, text_embed, null_text_embeds)       # type: ignore

        # mask out image embeddings with null image embeddings

        null_image_embed = self.null_image_embed.to(image_embed.dtype)

        image_embed = torch.where(image_keep_mask, image_embed, null_image_embed)

        # whether text embedding is used for conditioning depends on whether text encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right

        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)

        time_embed = self.to_time_embeds(diffusion_timesteps)

        learned_queries = repeat(self.learned_query, "d -> b 1 d", b=batch)

        if self.self_cond:
            learned_queries = torch.cat((self_cond, learned_queries), dim=-2)       # type: ignore

        input_stack = [text_embed, time_embed, image_embed, learned_queries]
        if text_encodings is not None:
            input_stack.insert(0, text_encodings)

        tokens = torch.cat(input_stack, dim=-2)

        # attend

        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)

        # Use the CLS token to predict the image embedding
        pred_image_embed = tokens[..., -1, :]

        return pred_image_embed


class BrainDiffusionPrior(DiffusionPrior):
    # Adapted from https://github.com/medarc-ai/fmri-reconstruction-nsd/blob/main/src/models.py#l232

    def __init__(
        self,
        config: BrainDiffusionPriorConfig,
        *args,
        **kwargs,
    ):

        self.config = config
        net = DiffusionPriorNetwork(
            dim=config.dim,
            num_timesteps=config.num_timesteps,
            num_time_embeds=config.num_time_embeds,
            num_image_embeds=config.num_image_embeds,
            num_text_embeds=config.num_text_embeds,
            max_text_len=config.max_text_len,
            self_cond=config.self_cond,
            depth=config.depth,
            num_output_tokens=config.num_output_tokens,
            rotary_emb=config.rotary_emb,
            normformer=config.normformer,
            norm_out=config.norm_out,
            dim_head=config.dim_head,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
        )

        super().__init__(
            net=net,
            image_embed_dim=config.image_embed_dim,
            loss_type=config.loss_type,
            cond_drop_prob=config.cond_drop_prob,
            image_cond_drop_prob=config.image_cond_drop_prob,
            condition_on_text_encodings=config.condition_on_text_encodings,
            image_size=config.image_size,
            predict_x_start=config.predict_x_start,
            sample_timesteps=config.sample_timesteps,
            beta_schedule=config.beta_schedule,
            clip=None,
            timesteps=config.num_timesteps,
            image_embed_scale=config.image_embed_scale,
            init_image_embed_l2norm=config.init_image_embed_l2norm,
            *args,
            **kwargs,
        )
        self.net = net

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_cond: dict | None = None,
        self_cond: torch.Tensor | None = None,
        clip_denoised: bool = True,
        cond_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        b = x.shape[0]
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=t,
            text_cond=text_cond,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            cond_scale=cond_scale,
        )
        noise = torch.randn(
            x.size(), device=x.device, dtype=x.dtype, generator=generator
        )
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(
        self,
        shape: torch.Size,
        brain_embedding: torch.Tensor | None = None,
        text_cond: dict = {},
        cond_scale: float = 1.0,
        generator: torch.Generator | None = None,
        progress_bar: bool = False,
        *args,
        **kwargs,
    ):
        batch = shape[0]
        device = cast(torch.device, self.device)

        if brain_embedding is not None:
            text_cond = {**text_cond, "text_embed": brain_embedding}

        image_embed = torch.randn(shape, device=device, generator=generator)

        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = image_embed.norm(dim=-1) * cast(float, self.image_embed_scale)

        for i in tqdm.tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc=f"DDPM sampling: {self.net.num_timesteps} timesteps",
            total=self.noise_scheduler.num_timesteps,
            disable=not progress_bar,
        ):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(
                image_embed,
                times,
                text_cond=text_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
                generator=generator,
            )

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(
        self,
        image_embed: torch.Tensor,
        times: torch.Tensor,
        text_cond: dict,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = noise if noise is not None else torch.randn_like(image_embed)

        image_embed_noisy = self.noise_scheduler.q_sample(
            x_start=image_embed, t=times, noise=noise
        )

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            image_cond_drop_prob=self.image_cond_drop_prob,
            **text_cond,
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            logging.info(f"Clamping pred with image scaling {self.image_embed_scale}")
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss, pred

    def forward(
        self,
        text: str | None = None,
        image: Image | None = None,
        brain_embedding: torch.Tensor | None = None,
        text_embedding: (
            torch.Tensor | None
        ) = None,  # allow for training on preprocessed CLIP text and image embeddings
        image_embedding: torch.Tensor | None = None,
        text_encodings: torch.Tensor | None = None,  # as well as CLIP text encodings
        times: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        # Validate inputs
        assert (
            (text is not None)
            or (text_embedding is not None)
            or (brain_embedding is not None)
        ), "either text, text embedding, or voxel must be supplied"

        if text is not None or image is not None:
            logging.warning(
                "Text or image was passed in. Using clip to embed text and image. This part of the code may not be tested properly."
            )
            assert (
                self.clip is not None
            ), "clip must be trained if you wish to pass in text or image"

        assert (image is not None) or (
            image_embedding is not None
        ), "either image or image embedding must be supplied"
        assert not (
            self.condition_on_text_encodings
            and (text_encodings is None and text is None)
        ), "text encodings must be present if you specified you wish to condition on it on initialization"

        if brain_embedding is not None:
            if text_embedding is not None:
                logging.warning(
                    "Both text embedding and brain embedding were passed in. Using brain embedding."
                )

            text_embedding = brain_embedding

        if image is not None:
            image_embedding, _ = self.clip.embed_image(image)  # type: ignore

        # calculate text conditionings, based on what is passed in

        if text is not None:
            text_embedding, text_encodings = self.clip.embed_text(text)  # type: ignore

        text_cond = dict(text_embed=text_embedding)

        if self.condition_on_text_encodings:
            assert (
                text_encodings is not None
            ), "text encodings must be present for diffusion prior if specified"
            text_cond = {**text_cond, "text_encodings": text_encodings}

        # Setup diffusion

        assert image_embedding is not None, "image embedding must be present"
        assert text_embedding is not None, "text embedding must be present"

        # timestep conditioning from ddpm

        batch, device = image_embedding.shape[0], image_embedding.device
        times = (
            times
            if times is not None
            else self.noise_scheduler.sample_random_times(batch)
        )

        image_embedding = image_embedding * cast(float, self.image_embed_scale)

        # calculate forward loss

        loss, pred = self.p_losses(
            image_embedding, times, text_cond=text_cond, *args, **kwargs
        )

        return loss, pred  # , text_embed


from dataclasses import dataclass

from typing import Any, Literal
from torch import nn
from torch import Tensor

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from brain_image.configs import get_device


def get_activation(name: str):
    match name:
        case "silu":
            return nn.SiLU
        case "elu":
            return nn.ELU
        case "gelu":
            return nn.GELU
        case "relu":
            return nn.ReLU
        case _:
            raise NotImplementedError(f"No activation function for {name}")


@dataclass
class DiffusionPriorConfig:
    d_input: int = 1024
    d_cond: int = 1024
    d_time: int = 512
    d_embed: int = 1024
    d_hidden_start: int = 1024
    d_hidden_scale: float = 0.5
    depth: int = 5
    act_func: Literal["silu", "elu", "gelu", "relu"] = "silu"
    dropout: float = 0.1

    cond_drop_prob: float = 0.1
    norm_scheme: Literal["z_scale", "l2_scale", "none"] = "none"

    num_training_timesteps: int = 1000


class SimpleDiffusionPrior(nn.Module):
    class EncoderLayer(nn.Module):
        def __init__(
            self,
            input_encoder: nn.Module,
            time_encoder: nn.Module,
            cond_encoder: nn.Module,
        ):
            super().__init__()
            self.input_encoder = input_encoder
            self.time_encoder = time_encoder
            self.cond_encoder = cond_encoder

        def forward(
            self,
            x,
            time_latent: Tensor,
            cond_latent: Tensor | None = None,
            cond_keep_mask: Tensor | None = None,
        ) -> Tensor:
            time_embed = self.time_encoder(time_latent)
            latent = x + time_embed

            if cond_latent is not None:
                cond_embed = self.cond_encoder(cond_latent)

                if cond_keep_mask is not None:
                    cond_embed = cond_embed * cond_keep_mask

                latent += cond_embed

            return self.input_encoder(latent)

    def __init__(self, config: DiffusionPriorConfig = DiffusionPriorConfig()):
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))
        self.config = config
        act_func = get_activation(config.act_func)

        self.time_proj = Timesteps(
            config.d_time, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.input_proj = nn.Sequential(
            nn.Linear(config.d_embed, config.d_hidden_start),
            nn.LayerNorm(config.d_hidden_start),
            act_func(),
        )

        encoder_layers = []
        decoder_layers = []

        self.hidden_dims = [
            int(config.d_hidden_start * config.d_hidden_scale**d)
            for d in range(config.depth)
        ]
        self.time_encoders = nn.ModuleList(
            [
                TimestepEmbedding(config.d_time, hidden_dim)
                for hidden_dim in self.hidden_dims
            ]
        )
        self.cond_encoders = nn.ModuleList(
            [nn.Linear(config.d_cond, hidden_dim) for hidden_dim in self.hidden_dims]
        )

        for d in range(config.depth - 1):
            hidden_dim = self.hidden_dims[d]
            next_hidden_dim = self.hidden_dims[d + 1]

            encode_layer = nn.Sequential(
                nn.Linear(hidden_dim, next_hidden_dim),
                nn.LayerNorm(next_hidden_dim),
                act_func(),
                nn.Dropout(config.dropout),
            )
            decode_layer = nn.Sequential(
                nn.Linear(next_hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_func(),
                nn.Dropout(config.dropout),
            )

            encoder_layers.append(
                self.EncoderLayer(
                    encode_layer, self.time_encoders[d], self.cond_encoders[d]
                )
            )
            decoder_layers.insert(
                0,
                self.EncoderLayer(
                    decode_layer, self.time_encoders[d + 1], self.cond_encoders[d + 1]
                ),
            )

        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)

        self.out_proj = nn.Linear(config.d_hidden_start, config.d_input)

        self.scheduler = DDPMScheduler(self.config.num_training_timesteps)

    def forward(
        self,
        x: Tensor,  # <B, D>
        timestep: torch.Tensor,  # <B> or scalar
        conditioning: torch.Tensor | None = None,  # <B, E>
        cond_drop_prob: float = 0,
    ) -> Tensor:
        x = self.input_proj(x)
        batch_size = x.size(0)

        cond_keep_mask = torch.rand(batch_size) >= cond_drop_prob
        cond_keep_mask = cond_keep_mask.unsqueeze(1).to(x.device)

        if timestep.dim() == 0:
            time_is_scalar = True
            timestep = timestep.view(1)
        elif timestep.size(0) != x.size(0):
            raise ValueError(
                f"Expected batch size {batch_size}, received timestep with batch size {timestep.size(0)}"
            )
        else:
            time_is_scalar = False

        if conditioning is not None and conditioning.size(0) != batch_size:
            raise ValueError(
                f"Expected batch size {batch_size}, received conditioning with batch size {conditioning.size(0)}"
            )

        t = self.time_proj(timestep)

        if time_is_scalar:
            t = t.expand(batch_size, -1)

        activations = []
        for encoder_layer in self.encoder_layers:
            activations.append(x)
            x = encoder_layer(x, t, conditioning, cond_keep_mask=cond_keep_mask)

        for decoder_layer, activation in zip(
            self.decoder_layers, reversed(activations)
        ):
            x = (
                decoder_layer(x, t, conditioning, cond_keep_mask=cond_keep_mask)
                + activation
            )

        x = self.out_proj(x)

        return x

    @torch.no_grad()
    def generate(
        self,
        num_steps: int | None = 50,
        guidance_scale: float = 5.0,
        conditioning: Tensor | None = None,
        batch_size: int | None = None,
        scheduler_timesteps: list[int] | None = None,
        scheduler_kwargs: dict[str, Any] = {},
        generator: torch.Generator | None = None,
        use_progress_bar: bool = False,
    ) -> Tensor:
        self.eval()

        device = self._dummy_param.device

        # Validate inputs
        if conditioning is None:
            if batch_size is None:
                raise ValueError(
                    f"Need to define either 'conditioning' or 'batch_size'"
                )

            latent_dim = self.config.d_embed

        else:
            batch_size = conditioning.size(0)
            latent_dim = conditioning.size(1)

            if latent_dim != self.config.d_embed:
                raise ValueError(
                    f"Expected conditioning with dim {self.config.d_embed}, received dim {latent_dim}"
                )

        if num_steps is None and scheduler_timesteps is None:
            raise ValueError(
                f"Need to define either 'num_steps' or 'scheduler_timesteps'"
            )

        if scheduler_timesteps is not None:
            num_steps = None

        # Prepare latents and timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_steps,
            timesteps=scheduler_timesteps,
            device=device,
            **scheduler_kwargs,
        )
        timesteps = self.scheduler.timesteps
        num_steps = len(timesteps)

        latent = torch.randn(batch_size, latent_dim, generator=generator, device=device)

        # Reverse diffusion
        for t in tqdm.tqdm(
            timesteps, desc="Denoising latent", disable=not use_progress_bar
        ):
            if conditioning is None or guidance_scale == 0:
                noise_pred = self.forward(latent, t)

            else:  # Classifier Free Guidance
                noise_pred_cond = self.forward(latent, t, conditioning)
                noise_pred_uncond = self.forward(latent, t)
                noise_pred = (
                    noise_pred_uncond
                    + (noise_pred_cond - noise_pred_uncond) * guidance_scale
                )

            latent = self.scheduler.step(
                noise_pred, int(t), latent, generator=generator
            ).prev_sample

        return latent

    def predict_step(self, noisy_latent: Tensor, timestep: Tensor, conditioning: Tensor | None = None, *args, **kwargs) -> Tensor:
        # x: <B, D>
        noisy_pred = self.forward(noisy_latent, timestep, conditioning, *args, **kwargs) # <B, D>

        prediction_type = cast(str, self.scheduler.config.prediction_type)   # type: ignore
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].unsqueeze(1).expand(-1, noisy_latent.size(1))  # <B, D>
        beta_prod_t = 1 - alpha_prod_t      # <B, 1>

        match prediction_type:
            case "epsilon":        
                clean_pred = (noisy_latent - beta_prod_t**0.5 * noisy_pred) / alpha_prod_t**0.5
            case "sample":
                clean_pred = noisy_pred
            case "v_prediction":
                clean_pred = (alpha_prod_t**0.5) * noisy_latent - (beta_prod_t**0.5) * noisy_pred

            case _:
                raise ValueError(
                    f"prediction_type given as {prediction_type} must be one of `epsilon`, `sample` or"
                    " `v_prediction`  for the DDPMScheduler."
                )

        return clean_pred
    
    def remove_noise(self, noisy_latent: Tensor, noise_pred: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        prediction_type = cast(str, self.scheduler.config.prediction_type)   # type: ignore
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].unsqueeze(1).expand(-1, noisy_latent.size(1))  # <B, D>
        beta_prod_t = 1 - alpha_prod_t      # <B, 1>

        match prediction_type:
            case "epsilon":        
                clean_pred = (noisy_latent - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
            case "sample":
                clean_pred = noise_pred
            case "v_prediction":
                clean_pred = (alpha_prod_t**0.5) * noisy_latent - (beta_prod_t**0.5) * noise_pred

            case _:
                raise ValueError(
                    f"prediction_type given as {prediction_type} must be one of `epsilon`, `sample` or"
                    " `v_prediction`  for the DDPMScheduler."
                )

        return clean_pred