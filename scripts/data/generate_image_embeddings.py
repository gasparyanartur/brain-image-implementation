import hydra
import logging
from pathlib import Path
from typing import Literal
from omegaconf import DictConfig
import torch
import tqdm
from torch.utils.data import DataLoader

from brain_image.configs import BaseConfig, GlobalConfig, get_device_str
from brain_image.data.datamodule import EEGDataModule
from brain_image.data.dataset.eeg_dataset import EEGDataset, EEGDatasetConfig
from brain_image.data.dataset.union import EEGDatasetConfigType
from brain_image.data.tensorcache import TensorCache
from brain_image.model.encoder.img_encoder.img_encoder import BaseImageEncoder
from brain_image.model.encoder.img_encoder.union import ImageEncoderName, load_image_encoder
from brain_image.utils import DTYPE, get_dtype, setup
from brain_image.data.io import batch_load_images


class EmbeddingGenerationConfig(BaseConfig):
    dataset: EEGDatasetConfigType
    model_names: list[ImageEncoderName] = ["clip_vith14", "aligned_synclr_vitb16", "unaligned_synclr_vitb16"]
    batch_size: int = 512
    splits: list[Literal["train", "test"]] = ["train", "test"]
    models_path: Path = Path("models")
    dtype: str = "float32"
    device: str | None = None
    compile: bool = True
    download_weights: bool = True
    cache_dir: Path = Path("tensorcache")

def run_generation(
    dataset: EEGDataset,
    output_dir: Path,
    split: Literal["train", "test"],
    batch_size: int,
    encoder: BaseImageEncoder,
    device: str | None = None,
) -> None:

    model_name = encoder.model_name
    logging.info(f"Generating {split} embeddings for model {model_name}")

    encoder_configs = [encoder.model_name, split]

    cache = TensorCache(cache_path=output_dir)
    logging.info(f"Saving embeddings with model configs {encoder_configs} to cache directory: {output_dir}")

    img_paths = dataset.get_image_paths()

    with torch.no_grad(), tqdm.tqdm(total=len(img_paths), desc="Generating embeddings...") as pbar:
        for i in range(0, len(img_paths), batch_size):
            paths = [img_paths[i] for i in range(i, min(i + batch_size, len(img_paths)))]
            imgs = batch_load_images(paths).to(device=device)

            latent = encoder.encode(imgs).detach().cpu()

            for i_path, path in enumerate(paths):
                save_path = cache.get_latent_path(path, *encoder_configs)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(latent[i_path], save_path)

            pbar.update(batch_size)

    logging.info(f"Finished generating {split} embeddings for model {model_name}")


def generate_all_embeddings(config: EmbeddingGenerationConfig) -> None:
    dataset_module = EEGDataModule(
        config.dataset
    )

    datasets = {
        split:  dataset_module.create_dataset(split, preload_cache=False, embeddings_to_compute_stats=[], compute_stats=False)
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
                dataset=datasets[split],
                output_dir=config.cache_dir,
                batch_size=config.batch_size,
                encoder=encoder,
                split=split,
                device=device,
            )


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="generate_image_embeddings",
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
