from typing import Literal
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from brain_image.model.encoder.eeg_encoder.eeg_encoder import (
    EEGEncoder,
    EEGEncoderConfig,
)
from brain_image.model.encoder.eeg_encoder.utils import EEGProjection, PatchEmbedding

# From https://github.com/ncclab-sustech/EEG_Image_decode/


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad_(False)

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]  # type: ignore


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad_(False)

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, d_model):
        super(SubjectEmbedding, self).__init__()
        self.subject_embedding = nn.Embedding(num_subjects, d_model)
        self.shared_embedding = nn.Parameter(
            torch.randn(1, d_model)
        )  # Shared token for unknown subjects
        self.mask_embedding = nn.Parameter(
            torch.randn(1, d_model)
        )  # Mask token embedding

    def forward(self, subject_ids):
        if subject_ids[0] is None or torch.any(
            subject_ids >= self.subject_embedding.num_embeddings
        ):
            batch_size = subject_ids.size(0)
            return self.shared_embedding.expand(batch_size, 1, -1)
        else:
            return self.subject_embedding(subject_ids).unsqueeze(1)


class DataEmbedding(nn.Module):
    def __init__(
        self,
        c_in,
        d_model,
        embed_type="fixed",
        freq="h",
        dropout=0.1,
        joint_train=False,
        num_subjects=None,
    ):
        super(DataEmbedding, self).__init__()
        if joint_train and num_subjects is not None:
            self.value_embedding = nn.ModuleDict(
                {
                    str(subject_id): nn.Linear(c_in, d_model)
                    for subject_id in range(num_subjects)
                }
            )
        else:
            self.value_embedding = nn.Linear(
                c_in, d_model
            )  # 如果没有指定subjects，则使用单一的value embedding

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.subject_embedding = (
            SubjectEmbedding(num_subjects, d_model)
            if num_subjects is not None
            else None
        )
        self.mask_token = nn.Parameter(torch.randn(1, d_model))  # Mask token embedding
        self.joint_train = joint_train

    def forward(self, x, x_mark, subject_ids=None, mask=None):
        if self.joint_train:
            # 使用针对每个subject的特定value embedding
            x = torch.stack([self.value_embedding[str(subject_id.item())](x[i]) for i, subject_id in enumerate(subject_ids)])  # type: ignore
        else:
            x = self.value_embedding(x)

        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark) + self.position_embedding(x)

        if mask is not None:
            x = x * (~mask.bool()) + self.mask_token * mask.float()

        if self.subject_embedding is not None:
            subject_emb = self.subject_embedding(
                subject_ids
            )  # (batch_size, 1, d_model)
            x = torch.cat(
                [subject_emb, x], dim=1
            )  # 在序列维度上拼接 (batch_size, seq_len + 1, d_model)

        return self.dropout(x)


class iTransformerConfig(BaseModel):
    task_name: str = "classification"  # Example task name
    seq_len: int = 250  # Sequence length
    pred_len: int = 250  # Prediction length
    num_channels: int = 63  # Number of EEG channel

    output_attention: bool = False  # Whether to output attention weights
    d_model: int = 250  # Model dimension
    embed: str = "timeF"  # Time encoding method
    freq: str = "h"  # Time frequency
    dropout: float = 0.25  # Dropout rate
    factor: int = 1  # Attention scaling factor
    n_heads: int = 4  # Number of attention heads
    e_layers: int = 1  # Number of encoder layers
    d_ff: int = 256  # Dimension of the feedforward network
    activation: str = "gelu"  # Activation function


class iTransformer(nn.Module):
    def __init__(
        self,
        config: iTransformerConfig,
        joint_train=False,
        num_subjects=None,
    ):
        super(iTransformer, self).__init__()
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.output_attention = config.output_attention

        self.enc_embedding = DataEmbedding(
            config.seq_len,
            config.d_model,
            config.embed,
            config.freq,
            config.dropout,
            joint_train=False,
            num_subjects=num_subjects,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, : self.config.num_channels, :]
        # print("enc_out", enc_out.shape)
        return enc_out


class AtmsEEGEncoderConfig(EEGEncoderConfig):
    eeg_encoder: Literal["atms"] = "atms"
    dropout: float = 0.5
    patch_out_size: int = 36
    hidden_dim: int = 1024
    flatten: bool = True


class AtmsEEGEncoder(EEGEncoder):
    def __init__(
        self,
        config: AtmsEEGEncoderConfig = AtmsEEGEncoderConfig(),
    ):
        super(AtmsEEGEncoder, self).__init__(config)
        self.config = config
        self.encoder = iTransformer(
            iTransformerConfig(seq_len=config.d_time, pred_len=config.d_time)
        )

        assert (
            config.d_channels is not None
        ), "d_channels must be specified for NiceEEGEncoder"

        self.patch_embedding = PatchEmbedding(
            d_embed=config.d_eeg,
            num_channels=config.d_channels,
            dropout=config.dropout,
        )

        self.proj = EEGProjection(
            d_input=config.patch_out_size * config.d_eeg,
            d_output=config.d_output,
            dropout=config.dropout,
        )

        if not config.flatten:
            raise NotImplementedError("Non-flattened case not implemented yet")

    def forward(self, x, sub: torch.Tensor | None = None):
        x = self.encoder(x, None, sub)

        x = self.patch_embedding(x)
        if x.size(1) != self.config.patch_out_size:
            raise ValueError(
                f"Expected patch_out_size {self.config.patch_out_size}, got {x.size(1)} for output of size {x.size()}. Please adjust the patch_out_size in the config."
            )

        x = x.flatten(start_dim=1)
        x = self.proj(x)

        # TODO: Handle non-flattened case

        return x
