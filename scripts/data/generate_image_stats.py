import logging
from pathlib import Path
from typing import Literal

import hydra
import torch
import tqdm
from omegaconf import DictConfig

from brain_image.configs import BaseConfig, GlobalConfig
from brain_image.data.data import get_from_batch
from brain_image.data.dataset.things_eeg2_dataset import ThingsEEG2DatasetConfig
from brain_image.data.tensorcache import TensorCache
from brain_image.model.encoder.img_encoder.union import IMAGE_ENCODER_DIM, ImageEncoderName
from brain_image.stats import IterativeStats
from brain_image.utils import setup


class ImageStatsConfig(BaseConfig):
    dataset: dict
    model_names: list[ImageEncoderName] = ["ip_sdxl_turbo_128"]
    batch_size: int = 32
    splits: list[Literal["train", "test"]] = ["train", "test"]
    cache_dir: Path = Path("tensorcache")
    stat_path: Path = Path("statistics")


def generate_stats(config: ImageStatsConfig) -> None:
    cache = TensorCache(config.cache_dir)
    dataset_config = ThingsEEG2DatasetConfig(**config.dataset)

    for split in config.splits:
        dataset = dataset_config.create_dataset(
            split,
            tensor_cache=cache,
            embeddings_key_to_name={name: name for name in config.model_names},
            compute_stats=False,
            preload_cache=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
        )
        eeg_stats = IterativeStats((dataset_config.num_channels, dataset_config.time_length))
        embedding_stats = {
            name: IterativeStats((IMAGE_ENCODER_DIM[name],)) for name in config.model_names
        }

        for batch in tqdm.tqdm(dataloader, desc=f"Processing {split} split"):
            eeg_stats.update(get_from_batch("eeg_data", batch, torch.Tensor))
            for name in config.model_names:
                embedding = get_from_batch(name, batch, torch.Tensor)
                embedding_stats[name].update(embedding.flatten(start_dim=1))

        stat_dir = config.stat_path / "datasets" / dataset_config.dataset / split
        stat_dir.mkdir(parents=True, exist_ok=True)
        eeg_stats.save_to_path(stat_dir / "eeg")
        for name, stats in embedding_stats.items():
            stats.save_to_path(stat_dir / name)
        logging.info("Saved image statistics for %s to %s", split, stat_dir)


@hydra.main(config_path=str(GlobalConfig.CONFIGS_DIR), config_name="generate_stats", version_base=None)
def main(cfg: DictConfig):
    setup()
    generate_stats(ImageStatsConfig.from_hydra_config(cfg))


if __name__ == "__main__":
    main()
