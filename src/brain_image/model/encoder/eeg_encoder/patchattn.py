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
    ):
        super().__init__()

        coords = get_channel_coords_from_names(channel_names, montage_type=montage_type)  # (C, 3)
        coords = coords / coords.norm(dim=1, keepdim=True)
        self.register_buffer("coord_embedding", coords, persistent=False)

        self.spatial_proj = nn.Sequential(
            nn.Linear(3, d_space_embed),
            nn.GELU(),
            nn.Linear(d_space_embed, d_space_embed),
        )

        self.temporal_embed = nn.Parameter(torch.randn(max_time_length, d_time_embed) * 0.02)

        self.embed_proj = nn.Sequential(
            nn.Linear(d_input + d_time_embed + d_space_embed, d_embed),
            nn.GELU(),
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, d_embed),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, d_input)
        B, C, T, F = x.size()

        # Spatial encoding from coordinates
        spatial_emb = self.spatial_proj(self.coord_embedding)  # (C, d_space_embed)
        spatial_emb = spatial_emb.unsqueeze(1).unsqueeze(0)  # (1, C, 1, d_space_embed)

        # Temporal encoding from learned embeddings
        time_emb = self.temporal_embed[:T]  # (T, d_time_embed)
        time_emb = time_emb.unsqueeze(0).unsqueeze(1)  # (1, 1, T, d_time_embed)

        # Concatenate input with spatiotemporal encoding
        emb = torch.cat(
            [x, spatial_emb.expand(B, -1, T, -1), time_emb.expand(B, C, -1, -1)], dim=-1
        )  # (B, C, T, d_input + d_space_embed + d_time_embed)

        emb = self.embed_proj(emb)  # (B, C, T, d_embed)
        return emb




class PatchAttentionLayer(nn.Module):
    def __init__(
        self,
        d_token: int,
        num_heads: int = 2,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,    
        ff_mult: int = 2,
    ):
        super().__init__()

        self.norm_c = nn.LayerNorm(d_token)
        self.attn_c = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm_t = nn.LayerNorm(d_token)
        self.attn_t = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm_ff = nn.LayerNorm(d_token)
        self.ff = nn.Sequential(
            nn.Linear(d_token, ff_mult * d_token),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_mult * d_token, d_token),
        )

        self.dropout = nn.Dropout(ff_dropout)
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, F)
        B, C, T, F = x.shape

        ## 1) Time Attention (per channel)
        x_t = x.reshape(B * C, T, F)  # (B*C, T, F)
        x_t_norm = self.norm_t(x_t)
        attn_out, _ = self.attn_t(x_t_norm, x_t_norm, x_t_norm)
        x = (x_t + attn_out).reshape(B, C, T, F)  # (B, C, T, F)

        # 2) Channel Attention (per timestep)
        x_c = x.permute(0, 2, 1, 3).contiguous().reshape(B * T, C, F)
        x_c_norm = self.norm_c(x_c)
        attn_out, _ = self.attn_c(x_c_norm, x_c_norm, x_c_norm)
        x = (x_c + attn_out).reshape(B, T, C, F).permute(0, 2, 1, 3)  # (B, C, T, F)
        x = x.contiguous()  # Ensure contiguous memory layout for the next operations

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
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        ff_mult: int = 1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                PatchAttentionLayer(
                    d_token=d_token,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(num_layers)
            ]
        )

        self.projection = nn.Linear(d_token, d_embed)

    def forward(self, x: Tensor, mean_pool: bool = True) -> Tensor:
        # x: (B, C, T, F)

        for layer in self.layers:
            x = layer(x)
        x = self.projection(x)  # (B, C, T, d_embed)

        if mean_pool:
            x = x.mean(dim=(1, 2))  # (B, d_embed)

        return x


class PatchAttentionEEGEncoderConfig(EEGEncoderConfig):
    eeg_encoder: Literal["patchattn"] = "patchattn"
    dropout: float = 0.2
    attn_dropout: float = 0.15
    ff_dropout: float = 0.25
    d_token: int = 256
    d_embed: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_mult: int = 2
    d_time_token: int = 128
    d_time_embed: int = 32
    d_space_embed: int = 128
    mean_pool: bool = True
    num_time_tokens: int = 8
    time_conv_norm: Literal["bn", "none"] = "bn"
    time_conv_dropout: float = 0.4

    flatten: bool = True


class PatchAttentionEEGEncoder(EEGEncoder):
    def __init__(self, config: PatchAttentionEEGEncoderConfig = PatchAttentionEEGEncoderConfig(), channel_names: list[str] | None = None, **kwargs):
        super(PatchAttentionEEGEncoder, self).__init__(config)
        if channel_names is None:
            logging.warning(
                "No channel names provided to PatchAttnEncoder. Using default 63 channel from Things-EEG2. This will not work correctly if you are using a different dataset."
            )
            channel_names = DEFAULT_CHANNEL_NAMES

        self.config = config

        assert config.d_channels is not None, "d_channels must be specified for PatchAttnEncoder"
        assert (
            len(channel_names) == config.d_channels
        ), f"Number of channel names {len(channel_names)} does not match d_channels {config.d_channels} in config."

        class ConvLayer(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride):
                super().__init__()
                _, k_t = kernel_size
                pad_t = (k_t - 1) // 2  # Same padding for time dimension

                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(0, pad_t), bias=False)
                match config.time_conv_norm:
                    case "bn":
                        self.norm = nn.BatchNorm2d(out_channels)
                    case "none":
                        self.norm = nn.Identity()
                    case _:
                        raise ValueError(f"Invalid time_conv_norm: {config.time_conv_norm}")
                self.act = nn.GELU()
                self.dropout = nn.Dropout(config.time_conv_dropout)

            def forward(self, x):
                x = self.conv(x)
                x = self.norm(x)
                x = self.act(x)
                x = self.dropout(x)
                return x

        # 250 -> ~30
        time_conv = nn.Sequential(
            ConvLayer(1, 64, kernel_size=(1, 25), stride=(1, 5)),  # 250 -> 50
            ConvLayer(64, 128, kernel_size=(1, 15), stride=(1, 5)),  # 50 -> 10
            ConvLayer(128, config.d_time_token, kernel_size=(1, 5), stride=(1, 1)),  # 10 -> 2
        )
        #time_conv = torch._dynamo.disable(
        #    time_conv, recursive=True, reason="Stride fails on the last batch, likely due to size mismatch. Needs further investigation."
        #)
        self.time_conv = time_conv

        self.channel_time_embed = ChannelTimeEmbedding(
            channel_names=channel_names,
            max_time_length=self.config.d_time,
            d_input=config.d_time_token,
            d_time_embed=config.d_time_embed,
            d_embed=config.d_token,
            d_space_embed=config.d_space_embed,
        )

        self.patch_attn_encoder = PatchAttentionEncoder(
            d_token=config.d_token,
            d_embed=config.d_embed,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            attn_dropout=config.attn_dropout,
            ff_mult=config.ff_mult,
            ff_dropout=config.ff_dropout,
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
        x = nn.functional.adaptive_avg_pool2d(
            x, (x.size(2), self.config.num_time_tokens)
        )  # <B, d_token, channel, 1>  # Mean pool over time dimension, keep channel dimension intact for spatial attention
        # Keep timedim for now for legacy, will remove soon
        x = x.permute(0, 2, 3, 1).contiguous()  # <B, channel, time_reduced, d_token>
        x = self.channel_time_embed(x)  # <B, channel, time, d_token>

        x = self.patch_attn_encoder(x, mean_pool=self.config.mean_pool)  # <B, d_token>
        x = self.proj(x)  # <B, d_output>
        return x
