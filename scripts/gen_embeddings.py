import hydra
import logging
from pathlib import Path
from typing import Literal
from omegaconf import DictConfig
import torch
import tqdm

from brain_image.configs import BaseConfig, GlobalConfig, get_device
from brain_image.data import (
    EEGDatasetConfig,
    batch_load_images,
    get_image_paths,
    preprocess_image,
)
from brain_image.model import load_image_encoder
from brain_image.utils import DTYPE, get_dtype


class EmbeddingGenerationConfig(BaseConfig):
    batch_size: int = 32
    models: list[str] = ["synclr", "aligned_synclr"]
    splits: list[Literal["train", "test"]] = ["train", "test"]
    img_size: tuple[int, int] = (224, 224)
    models_path: Path = Path("models")
    dtype: str = "float16"
    device: str | None = None
    data_config: EEGDatasetConfig = EEGDatasetConfig()

    output_dir: Path | None = (
        None  # If None, will be the same as the data_config.data_path / data_config.latents_dir
    )


def generate_latents(
    embed_model: torch.nn.Module,
    img_paths: list[Path],
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    device: torch.device = get_device(),
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    """Generate embeddings for a given split of images."""
    embed_model.eval()
    embed_model.to(device=device, dtype=dtype)
    embed_model.requires_grad_(False)

    latents = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(img_paths), batch_size)):
            paths = img_paths[i : i + batch_size]
            imgs = batch_load_images(paths).to(device, dtype=dtype)
            imgs = preprocess_image(imgs, img_size=img_size)

            latent = embed_model(imgs).detach().cpu()
            latents.append(latent)

    return torch.concat(latents, dim=0)


def run_generation(
    img_dir: Path,
    output_dir: Path,
    model_name: str,
    split: Literal["train", "test"],
    models_path: Path = Path("models"),
    batch_size: int = 512,
    img_size: tuple[int, int] = (224, 224),
    device: torch.device = get_device(),
    dtype: torch.dtype = DTYPE,
) -> None:
    """Run the embedding generation process."""

    image_encoder = load_image_encoder(model_name, models_path=models_path)
    logging.info(f"Generating {split} embeddings for model {model_name}")

    img_paths = get_image_paths(
        img_dir,
        split=split,
    )

    train_embeddings = generate_latents(
        image_encoder,
        img_paths,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        dtype=dtype,
    )

    dst_dir = output_dir / model_name / f"{split}_embeddings.pt"
    dst_dir.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving {split} embeddings to {dst_dir}")
    torch.save(train_embeddings, dst_dir)


def generate_all_embeddings(config: EmbeddingGenerationConfig) -> None:
    img_dir = config.data_config.data_path / config.data_config.imgs_dir
    embed_dir = (
        config.output_dir
        or config.data_config.data_path / config.data_config.latents_dir
    )

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory {img_dir} does not exist")

    if not embed_dir.exists():
        embed_dir.mkdir(parents=True, exist_ok=True)

    for split in config.splits:
        for model_name in config.models:
            run_generation(
                img_dir,
                embed_dir,
                batch_size=config.batch_size,
                model_name=model_name,
                split=split,
                models_path=config.models_path,
                img_size=config.img_size,
                device=torch.device(config.device) if config.device else get_device(),
                dtype=get_dtype(config.dtype),
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

    # Create config from the composed configuration
    from brain_image.data import EEGDatasetConfig

    dataset_config = EEGDatasetConfig(**cfg.dataset)

    # Create the embedding generation config
    config = EmbeddingGenerationConfig(
        batch_size=cfg.batch_size,
        models=cfg.models,
        splits=cfg.splits,
        img_size=tuple(cfg.img_size),
        models_path=Path(cfg.models_path),
        dtype=cfg.dtype,
        device=cfg.device,
        data_config=dataset_config,
        output_dir=Path(cfg.output_dir) if cfg.output_dir else None,
    )

    logging.info("Starting embedding generation")
    logging.info(f"Config: {config}")
    generate_all_embeddings(config)


if __name__ == "__main__":
    main()
