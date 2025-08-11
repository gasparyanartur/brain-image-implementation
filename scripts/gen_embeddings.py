from collections.abc import Callable
import hydra
import logging
from pathlib import Path
from typing import Literal
from omegaconf import DictConfig
import torch
import tqdm

from brain_image.configs import BaseConfig, GlobalConfig, get_device_str
from brain_image.data import (
    EEGDatasetConfig,
    TensorCache,
    batch_load_images,
    get_image_paths,
    preprocess_image,
)
from brain_image.model import load_image_encoder
from brain_image.utils import DTYPE, get_dtype
from dreamsim.model import PerceptualModel


class EmbeddingGenerationConfig(BaseConfig):
    model_name: Literal["aligned_synclr_16", "unaligned_synclr_16", "aligned_clip_32", "unaligned_clip_32", "sd_highlevel", "sd_lowlevel"]
    task_type: Literal["align", "recon"]

    batch_size: int = 512
    splits: list[Literal["train", "test"]] = ["train", "test"]
    img_size: tuple[int, int] = (224, 224)
    models_path: Path = Path("models")
    dtype: str = "float32"
    device: str | None = None
    download_weights: bool = True
    img_dir: Path = Path("data/things-eeg2/imgs")
    output_dir: Path = Path("cache/tensorcache")

def run_generation(
    img_dir: Path,
    output_dir: Path,
    task_type: str,
    model_name: str,
    split: Literal["train", "test"],
    models_path: Path = Path("models"),
    batch_size: int = 512,
    img_size: tuple[int, int] = (224, 224),
    device: str | None = None,
    dtype: torch.dtype = DTYPE,
    download_weights: bool = True,
) -> None:
    """Run the embedding generation process."""

    logging.info(f"Loading image encoder for model {model_name} on device {device}")
    image_encoder = load_image_encoder(
        task_type,
        model_name,
        models_path=models_path,
        download_weights=download_weights,
        device=device,
        img_size=img_size,
        dtype=dtype,
    )
    logging.info(f"Generating {split} embeddings for model {model_name}")

    img_paths = get_image_paths(
        img_dir,
        split=split,
    )

    encoder_configs = [task_type, model_name, split]

    device = device or get_device_str()

    cache = TensorCache(cache_path=output_dir)
    logging.info(f"Saving embeddings with model configs {encoder_configs} to cache directory: {output_dir}")

    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(img_paths), batch_size), desc="Generating embeddings..."):
            paths = img_paths[i : i + batch_size]
            imgs = batch_load_images(paths).to(device=device)

            latent = image_encoder(imgs)
            
            for i_path, path in enumerate(paths):
                cache.save(latent[i_path], str(path), *encoder_configs)


    logging.info(f"Finished generating {split} embeddings for model {model_name}")


def generate_all_embeddings(config: EmbeddingGenerationConfig) -> None:
    device = config.device or get_device_str()
    logging.info(f"Generating all embeddings using device: {device}")

    for split in config.splits:
            run_generation(
                config.img_dir,
                output_dir=config.output_dir,
                task_type=config.task_type,
                batch_size=config.batch_size,
                model_name=config.model_name,
                split=split,
                models_path=config.models_path,
                img_size=config.img_size,
                device=device,
                dtype=get_dtype(config.dtype),
                download_weights=config.download_weights,
            )


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="gen_embeddings",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main function for embedding generation with clean configuration."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Using device: {get_device_str()}")

    # Create the embedding generation config
    config = EmbeddingGenerationConfig(
        batch_size=cfg.batch_size,
        model_name=cfg.model_name,
        task_type=cfg.task_type,
        splits=cfg.splits,
        img_size=tuple(cfg.img_size),
        models_path=Path(cfg.models_path),
        dtype=cfg.dtype,
        device=cfg.device,
        img_dir=Path(cfg.img_dir),
        output_dir=Path(cfg.output_dir),
        download_weights=cfg.download_weights,
    )

    logging.info("Starting embedding generation")
    logging.info(f"Config: {config}")
    generate_all_embeddings(config)


if __name__ == "__main__":
    main()
