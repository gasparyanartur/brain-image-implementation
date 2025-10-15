import hydra
import logging
from pathlib import Path
from typing import Literal
from omegaconf import DictConfig
import torch
import tqdm

from brain_image.configs import BaseConfig, GlobalConfig, get_device_str
from brain_image.data import (
    TensorCache,
    batch_load_images,
    get_image_paths,
)
from brain_image.model.img_encoder import IMAGE_ENCODER, load_image_encoder
from brain_image.utils import DTYPE, get_dtype, setup


class EmbeddingGenerationConfig(BaseConfig):
    model_names: list[IMAGE_ENCODER] = ["clip_vith14", "aligned_synclr_vitb16", "unaligned_synclr_vitb16"]
    batch_size: int = 512
    splits: list[Literal["train", "test"]] = ["train", "test"]
    models_path: Path = Path("models")
    dtype: str = "float32"
    device: str | None = None
    compile: bool = True
    download_weights: bool = True
    img_dir: Path = Path("data/things-eeg2/imgs")
    output_dir: Path = Path("tensorcache")

def run_generation(
    img_dir: Path,
    output_dir: Path,
    model_name: IMAGE_ENCODER,
    split: Literal["train", "test"],
    models_path: Path = Path("models"),
    batch_size: int = 512,
    device: str | None = None,
    dtype: torch.dtype = DTYPE,
    download_weights: bool = True,
    compile_model: bool = True
) -> None:
    logging.info(f"Loading image encoder for model {model_name} on device {device}")
    image_encoder = load_image_encoder(
        model_name,
        models_path=models_path,
        download_weights=download_weights,
        device=device,
        dtype=dtype,
        compile=compile_model
    )
    logging.info(f"Generating {split} embeddings for model {model_name}")

    img_paths = get_image_paths(
        img_dir,
        split=split,
    )

    encoder_configs = [model_name, split]

    cache = TensorCache(cache_path=output_dir)
    logging.info(f"Saving embeddings with model configs {encoder_configs} to cache directory: {output_dir}")

    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(img_paths), batch_size), desc="Generating embeddings..."):
            paths = img_paths[i : i + batch_size]
            if len(paths) < batch_size:
                # pad with arbitrary img_path, remove later. This is to make sure all batch sizes are the same for the compiled model.
                num_pad = batch_size - len(paths)
                paths += [paths[-1]] * num_pad
            else:
                num_pad = 0
                
            imgs = batch_load_images(paths).to(device=device)

            latent = image_encoder.encode(imgs).detach().cpu()

            if num_pad > 0:
                # Remove padding
                latent = latent[:-num_pad]    
                paths = paths[:-num_pad]
            
            for i_path, path in enumerate(paths):
                cache.save(latent[i_path], str(path), *encoder_configs)


    logging.info(f"Finished generating {split} embeddings for model {model_name}")


def generate_all_embeddings(config: EmbeddingGenerationConfig) -> None:
    device = config.device or get_device_str()
    logging.info(f"Generating all embeddings using device: {device}")

    for model_name in config.model_names:
        logging.info(f"Generating embeddings for model {model_name}"))

        for split in config.splits:
                run_generation(
                    config.img_dir,
                    output_dir=config.output_dir,
                    batch_size=config.batch_size,
                    model_name=model_name,
                    split=split,
                    models_path=config.models_path,
                    device=device,
                    dtype=get_dtype(config.dtype),
                    download_weights=config.download_weights,
                    compile_model=config.compile
                )


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="generate_embeddings",
    version_base=None,
)
def main(cfg: DictConfig):
    setup()

    config = EmbeddingGenerationConfig.from_hydra_config(cfg)
    logging.info("Starting embedding generation")
    for key, value in config.model_dump(mode="json").items():
        logging.info(f"{key}: {value}")

    generate_all_embeddings(config)


if __name__ == "__main__":
    main()
