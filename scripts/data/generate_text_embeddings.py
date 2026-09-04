from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import hydra
import torch
import tqdm
from omegaconf import DictConfig

from brain_image.configs import BaseConfig, GlobalConfig, get_device_str
from brain_image.data.io import get_image_paths
from brain_image.data.tensorcache import TensorCache
from brain_image.model.encoder.text_encoder.text_encoder import BaseTextEncoder
from brain_image.model.encoder.text_encoder.union import TextEncoderName, load_text_encoder
from brain_image.utils import get_dtype, setup


class TextEmbeddingGenerationConfig(BaseConfig):
    dataset: dict
    model_names: list[TextEncoderName] = ["t5_base", "clip_vitl14_text"]
    caption_path: Path = Path("data/things-eeg2/captions/local.jsonl")
    batch_size: int = 256
    splits: list[Literal["train", "test"]] = ["train", "test"]
    dtype: str = "float32"
    device: str | None = None
    compile: bool = False
    cache_dir: Path = Path("tensorcache")


def load_captions(caption_path: Path) -> dict[str, str]:
    """Load captions from a JSONL file into a {img_path: caption} dict."""
    captions: dict[str, str] = {}
    with open(caption_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                captions[entry["path"]] = entry["caption"]
            except (json.JSONDecodeError, KeyError):
                continue
    return captions


def run_text_generation(
    img_paths: list[Path],
    captions: dict[str, str],
    output_dir: Path,
    split: Literal["train", "test"],
    batch_size: int,
    encoder: BaseTextEncoder,
    device: str | None = None,
) -> None:
    model_name = encoder.model_name
    logging.info(f"Generating '{split}' text embeddings for model '{model_name}'")

    encoder_configs = [model_name, split]
    cache = TensorCache(cache_path=output_dir)
    logging.info(f"Saving text embeddings with configs {encoder_configs} to: {output_dir}")

    missing = [p for p in img_paths if str(p) not in captions]
    if missing:
        logging.warning(f"{len(missing)} image paths have no caption and will be skipped.")
    paths_with_captions = [p for p in img_paths if str(p) in captions]

    with torch.no_grad(), tqdm.tqdm(total=len(paths_with_captions), desc=f"[{split}] {model_name}") as pbar:
        for i in range(0, len(paths_with_captions), batch_size):
            batch_paths = paths_with_captions[i : i + batch_size]
            batch_texts = [captions[str(p)] for p in batch_paths]

            embeddings = encoder.encode(batch_texts).detach().cpu()

            for j, path in enumerate(batch_paths):
                save_path = cache.get_latent_path(path, *encoder_configs)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(embeddings[j], save_path)

            pbar.update(len(batch_paths))

    logging.info(f"Finished generating '{split}' text embeddings for model '{model_name}'")


def generate_all_text_embeddings(config: TextEmbeddingGenerationConfig) -> None:
    captions = load_captions(config.caption_path)
    logging.info(f"Loaded {len(captions)} captions from {config.caption_path}")

    device = config.device or get_device_str()
    logging.info(f"Using device: {device}")

    for model_name in config.model_names:
        encoder = load_text_encoder(
            model_name,
            device=device,
            dtype=get_dtype(config.dtype),
            compile=config.compile,
        )

        for split in config.splits:
            image_dir = Path(config.dataset["data_path"]) / config.dataset.get("img_dir", "imgs")
            img_paths = get_image_paths(image_dir, split, extensions=(".jpg",))
            run_text_generation(
                img_paths=img_paths,
                captions=captions,
                output_dir=config.cache_dir,
                split=split,
                batch_size=config.batch_size,
                encoder=encoder,
                device=device,
            )

        # Free memory between models
        del encoder
        torch.cuda.empty_cache()


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="generate_text_embeddings",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    setup()

    config = TextEmbeddingGenerationConfig.from_hydra_config(cfg)
    logging.info("Starting text embedding generation")
    for key, value in config.model_dump(mode="json").items():
        logging.info(f"  {key}: {value}")

    generate_all_text_embeddings(config)


if __name__ == "__main__":
    main()
