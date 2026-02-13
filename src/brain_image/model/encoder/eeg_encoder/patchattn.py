import logging
from typing import Literal

import torch
from brain_image.data.data import get_channel_coords_from_names
from brain_image.data.dataset.defaults import DEFAULT_CHANNEL_NAMES
from brain_image.model.encoder.eeg_encoder.eeg_encoder import EEGEncoder, EEGEncoderConfig

from torch import Tensor, nn


class ChannelTimeEmbedding(nn.Module):
    def __init__(
        self,
        channel_names: list[str],
        max_time_length: int,
        d_time_embed: int = 64,
        d_space_embed: int = 64,
        d_input: int = 64,
        d_embed: int = 192,
        montage_type: str = "standard_1020",
        spatial_fourier_freqs: int = 4,  # 0 to disable Fourier features
    ):
        super().__init__()

        coords = get_channel_coords_from_names(channel_names, montage_type=montage_type)  # (C, 3)
        coords = coords / coords.norm(dim=1, keepdim=True)

        self.use_fourier = spatial_fourier_freqs > 0
        self.spatial_fourier_freqs = spatial_fourier_freqs

        self.coord_proj = nn.Linear(3 * (2 * spatial_fourier_freqs) if self.use_fourier else 3, d_space_embed)

        time_embed = self._build_time_embedding(max_time_length, d_time_embed)
        coord_embed = self._build_coord_embedding(coords)

        self.embed_proj = nn.Sequential(
            nn.Linear(d_input + d_time_embed + d_space_embed, d_embed),
            nn.GELU(),
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, d_embed),
        )

        self.register_buffer("time_embedding", time_embed, persistent=False)
        self.register_buffer("coord_embedding", coord_embed, persistent=False)
   
   
    def _fourier_encode(self, coords: Tensor) -> Tensor:
        # coords: (C, 3)
        spectral_freqs = 2 ** torch.arange(self.spatial_fourier_freqs) 
        coords = coords.unsqueeze(-1) * spectral_freqs  # (C, 3, F)
        coords = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        return coords.flatten(1)  # (C, 3 * 2 * F)

    def _build_coord_embedding(self, coords):
        if self.use_fourier:
            coords = self._fourier_encode(coords)
        return coords  # (C, F)
    
    def _build_time_embedding(self, T, dim):
        pos = torch.arange(T).unsqueeze(1)
        i = torch.arange(dim // 2).unsqueeze(0)
        angle = pos / (10000 ** (2 * i / dim))
        emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
        return emb  # (T, F)
    

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, F, C, T)
        B, C, T, F = x.size()
        spatial_emb = self.coord_proj(self.coord_embedding)  # (C, F)
        spatial_emb = spatial_emb.unsqueeze(1).unsqueeze(0) # (1, C, 1, F)

        time_emb = self.time_embedding[:T]  # (T, F), # type: ignore
        time_emb = time_emb.unsqueeze(0).unsqueeze(1)  # (1, 1, T, F) #type: ignore

        emb = torch.cat([x, spatial_emb.expand(B, -1, T, -1), time_emb.expand(B, C, -1, -1)], dim=-1)  # (B, C, T, F + F + F)
        emb = self.embed_proj(emb)  # (B, C, T, d_embed)
        return emb


class PatchAttentionLayer(nn.Module):
    def __init__(
        self,
        d_token: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 2,
    ):
        super().__init__()

        self.norm_c = nn.LayerNorm(d_token)
        self.attn_c = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_t = nn.LayerNorm(d_token)
        self.attn_t = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ff = nn.LayerNorm(d_token)
        self.ff = nn.Sequential(
            nn.Linear(d_token, ff_mult * d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_token, d_token),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, F)
        B, C, T, F = x.shape

        # 1) Channel Attention (per timestep)
        x_c = x.permute(0, 2, 1, 3).reshape(B * T, C, F).contiguous()

        x_c_norm = self.norm_c(x_c)
        attn_out, _ = self.attn_c(x_c_norm, x_c_norm, x_c_norm)

        x_c = x_c + attn_out

        x = x_c.reshape(B, T, C, F).permute(0, 2, 1, 3)  

        # 2) Time Attention (per channel)
        x_t = x.reshape(B * C, T, F).contiguous()

        x_t_norm = self.norm_t(x_t)
        attn_out, _ = self.attn_t(x_t_norm, x_t_norm, x_t_norm)

        x_t = x_t + attn_out

        x = x_t.reshape(B, C, T, F)

        # 3) Feed Forward
        x_ff = self.norm_ff(x)
        x_ff = self.ff(x_ff)

        x = x + self.dropout(x_ff)

        return x


class PatchAttentionEncoder(nn.Module):
    def __init__(
        self,
        d_token: int,
        d_embed: int = 256,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                PatchAttentionLayer(
                    d_token=d_token,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(num_layers)
            ]
        )

        self.projection = nn.Linear(d_token, d_embed)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, F)

        for layer in self.layers:
            x = layer(x)


        x = self.projection(x)  # (B, C, T, d_embed)

        # Mean pool over channels and time
        x = x.mean(dim=[1, 2])  # (B, d_embed)

        return x


class PatchAttentionEEGEncoderConfig(EEGEncoderConfig):
    eeg_encoder: Literal["patchattn"] = "patchattn"
    dropout: float = 0.5
    attn_dropout: float = 0.1
    d_token: int = 128
    d_embed: int = 512
    time_kernel_size: int = 25
    time_stride: int = 5
    num_layers: int = 3
    num_heads: int = 2
    ff_mult: int = 2
    d_time_token: int = 64
    d_time_embed: int = 64
    d_space_embed: int = 64
    num_spatial_fourier_freqs: int = 32  # 0 to disable Fourier features

    flatten: bool = True

class PatchAttentionEEGEncoder(EEGEncoder):
    def __init__(
        self,
        config: PatchAttentionEEGEncoderConfig = PatchAttentionEEGEncoderConfig(),
        channel_names: list[str] | None = None,
        **kwargs
    ):
        super(PatchAttentionEEGEncoder, self).__init__(config)
        if channel_names is None:
            logging.warning("No channel names provided to PatchAttnEncoder. Using default 63 channel from Things-EEG2. This will not work correctly if you are using a different dataset.")
            channel_names = DEFAULT_CHANNEL_NAMES

        self.config = config

        assert config.d_channels is not None, "d_channels must be specified for PatchAttnEncoder"
        assert len(channel_names) == config.d_channels, f"Number of channel names {len(channel_names)} does not match d_channels {config.d_channels} in config."


        # 250 -> ~30
        self.time_conv = torch._dynamo.disable(nn.Sequential(
            nn.Conv2d(1, config.d_time_token, kernel_size=(1, config.time_kernel_size), stride=(1, 5)),
        ), recursive=True, reason="Stride fails on the last batch, likely due to size mismatch. Needs further investigation.")


        self.channel_time_embed = ChannelTimeEmbedding(
            channel_names=channel_names,
            max_time_length=self.config.d_time,
            d_input=config.d_time_token,
            d_time_embed=config.d_time_embed,
            d_embed=config.d_token,
            d_space_embed=config.d_space_embed,
            spatial_fourier_freqs=config.num_spatial_fourier_freqs,
        )

        self.patch_attn_encoder = PatchAttentionEncoder(
            d_token=config.d_token,
            d_embed=config.d_embed,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.attn_dropout,
            ff_mult=config.ff_mult,
        )
        
        self.proj = nn.Sequential(
            nn.LayerNorm(config.d_embed),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_embed, config.d_output),
            nn.LayerNorm(config.d_output),
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # <B, channel, time>
        x = x.unsqueeze(1).contiguous()  # <B, 1, channel, time>
        x = self.time_conv(x)  # <B, d_token, channel, time_reduced>
        x = x.permute(0, 2, 3, 1).contiguous()  # <B, channel, time_reduced, d_token>
        x = self.channel_time_embed(x)  # <B, channel, time, d_token>

        x = self.patch_attn_encoder(x)  # <B, d_token>
        x = self.proj(x) # <B, d_output>
        return x