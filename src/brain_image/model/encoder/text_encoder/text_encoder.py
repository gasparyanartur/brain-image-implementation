from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class BaseTextEncoder(nn.Module):
    """Base class for all text encoders."""

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__()
        self.model_name = model_name

    def tokenize(self, text: list[str]) -> dict:
        raise NotImplementedError

    def encode(self, tokens: Tensor | list[str] | dict) -> Tensor:
        raise NotImplementedError

    def forward(self, tokens: Tensor | list[str] | dict) -> Tensor:
        return self.encode(tokens)


class T5TextEncoder(BaseTextEncoder):
    def __init__(self, model_name: str = "t5-base", *args, **kwargs):
        super().__init__(model_name=model_name)
        from transformers import T5EncoderModel, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.requires_grad_(False)

    def tokenize(self, text: list[str]) -> dict:
        out = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return {
            "input_ids": out.input_ids,
            "attention_mask": out.attention_mask,
        }

    def mean_pool(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (last_hidden_state * mask).sum(dim=1)  # (B, D)
        denom = mask.sum(dim=1).clamp_min(1e-6)  # (B, 1)
        return summed / denom

    def get_final_embedding(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        return self.mean_pool(last_hidden_state, attention_mask)

    def encode(self, tokens: Tensor | list[str] | dict) -> Tensor:
        if isinstance(tokens, list):
            tokens = self.tokenize(tokens)

        device = next(self.model.parameters()).device
        with torch.no_grad():
            if isinstance(tokens, dict):
                tokens = {k: v.to(device) for k, v in tokens.items()}
                outputs = self.model(**tokens, return_dict=True)
            else:
                tokens = tokens.to(device)
                outputs = self.model(input_ids=tokens, return_dict=True)
                tokens = {"attention_mask": torch.ones(tokens.shape[:2], device=device)}

        return self.get_final_embedding(outputs.last_hidden_state, tokens["attention_mask"])


class CLIPTextEncoder(BaseTextEncoder):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", *args, **kwargs):
        super().__init__(model_name=model_name)
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.requires_grad_(False)

    def tokenize(self, text: list[str]) -> dict:
        out = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return {
            "input_ids": out.input_ids,
            "attention_mask": out.attention_mask,
        }

    def encode(self, tokens: Tensor | list[str] | dict) -> Tensor:
        if isinstance(tokens, list):
            tokens = self.tokenize(tokens)

        device = next(self.model.parameters()).device
        with torch.no_grad():
            if isinstance(tokens, dict):
                tokens = {k: v.to(device) for k, v in tokens.items()}
                outputs = self.model(**tokens, return_dict=True)
            else:
                tokens = tokens.to(device)
                outputs = self.model(input_ids=tokens, return_dict=True)

        return outputs.pooler_output


class GemmaTextEncoder(BaseTextEncoder):
    def __init__(self, model_name: str = "google/embeddinggemma-300m", *args, **kwargs):
        super().__init__(model_name=model_name)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model.requires_grad_(False)

    def tokenize(self, text: list[str]) -> dict:
        raise NotImplementedError("GemmaTextEncoder uses SentenceTransformer; call encode() directly with text.")

    def encode(self, tokens: Tensor | list[str] | dict) -> Tensor:
        if not isinstance(tokens, list):
            raise TypeError("GemmaTextEncoder.encode() expects a list[str].")
        embeddings = self.model.encode_document(tokens, convert_to_numpy=True)
        return torch.tensor(embeddings)
