import hydra
import logging
from pathlib import Path
from typing import Literal
from omegaconf import DictConfig
import torch
import tqdm
from torch.utils.data import DataLoader

from brain_image.configs import BaseConfig, GlobalConfig, get_device_str
from brain_image.model.img_encoder import IMAGE_ENCODER, BaseImageEncoder, load_image_encoder
from brain_image.utils import DTYPE, get_dtype, setup
from brain_image.data.data import EEGDataModule, EEGDatasetConfig, TensorCache, batch_load_images


class EmbeddingGenerationConfig(BaseConfig):
    dataset: EEGDatasetConfig
    model_names: list[IMAGE_ENCODER] = ["clip_vith14", "aligned_synclr_vitb16", "unaligned_synclr_vitb16"]
    batch_size: int = 512
    splits: list[Literal["train", "test"]] = ["train", "test"]
    models_path: Path = Path("models")
    dtype: str = "float32"
    device: str | None = None
    compile: bool = True
    download_weights: bool = True
    cache_dir: Path = Path("tensorcache")

def run_generation(
    dataloader: DataLoader,
    output_dir: Path,
    split: Literal["train", "test"],
    encoder: BaseImageEncoder,
    device: str | None = None,
) -> None:

    model_name = encoder.model_name
    logging.info(f"Generating {split} embeddings for model {model_name}")

    encoder_configs = [encoder.model_name, split]

    cache = TensorCache(cache_path=output_dir)
    logging.info(f"Saving embeddings with model configs {encoder_configs} to cache directory: {output_dir}")

    with torch.no_grad(), tqdm.tqdm(total=len(dataloader), desc="Generating embeddings...") as pbar:
            for batch in dataloader:
                paths = batch["img_path"]
                imgs = batch_load_images(paths).to(device=device)

                latent = encoder.encode(imgs).detach().cpu()

                for i_path, path in enumerate(paths):
                    cache.save(latent[i_path], str(path), *encoder_configs)

    logging.info(f"Finished generating {split} embeddings for model {model_name}")


def generate_all_embeddings(config: EmbeddingGenerationConfig) -> None:
    dataset_module = EEGDataModule(
        config.dataset
    )

    dataloaders = {
        split:  dataset_module.create_dataloader("train", shuffle=False)
        for split in config.splits
    }

    device = config.device or get_device_str()
    logging.info(f"Generating all embeddings using device: {device}")

    for model_name in config.model_names:
        encoder = load_image_encoder(
            model_name,
            models_path=config.models_path,
            download_weights=config.download_weights,
            device=device,
            dtype=get_dtype(config.dtype),
            compile=config.compile
        )

        for split in config.splits:
            run_generation(
                dataloader=dataloaders[split],
                output_dir=config.cache_dir,
                encoder=encoder,
                split=split,
                device=device,
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
