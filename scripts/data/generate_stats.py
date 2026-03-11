# Dataset info:
# train_mean
# train_std

import logging
from pathlib import Path
from typing import Literal

import hydra
from omegaconf import DictConfig
import torch
import tqdm

from brain_image.configs import BaseConfig, GlobalConfig
from brain_image.data.data import get_from_batch
from brain_image.data.datamodule import EEGDataModule
from brain_image.data.dataset.eeg_dataset import EEGDatasetConfig
from brain_image.data.dataset.union import EEGDatasetConfigType
from brain_image.data.tensorcache import TensorCache
from brain_image.model.encoder.encoder import ALIGN_ENCODER_DIM, AlignEncoderName
from brain_image.stats import IterativeStats
from brain_image.utils import setup

from torch import Tensor


class GetDataInformationConfig(BaseConfig):
    dataset: EEGDatasetConfigType
    model_names: list[AlignEncoderName] = ["clip_vith14", "aligned_synclr_vitb16", "unaligned_synclr_vitb16"]
    batch_size: int = 512
    splits: list[Literal["train", "test"]] = ["train", "test"]
    cache_dir: Path = Path("tensorcache")
    stat_path: Path = Path("statistics")


def get_data_information(config: GetDataInformationConfig):
    cache = TensorCache(config.cache_dir)

    # Initialise without embeddings so the module doesn't try to load stats
    # that don't exist yet (we are the ones generating them).
    data = EEGDataModule(config.dataset, tensor_cache=cache)

    for split in config.splits:
        # Bypass EEGDataModule.create_dataset to avoid the duplicate-kwarg issue
        # and to pass compute_stats=False (stats don't exist yet).
        dataset = config.dataset.create_dataset(
            split,
            tensor_cache=cache,
            embeddings_key_to_name={model_name: model_name for model_name in config.model_names},
            compute_stats=False,
            preload_cache=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=min(8, __import__("multiprocessing").cpu_count()),
        )
        eeg_shape = (data.config.num_channels, data.config.time_length)

        eeg_stats = IterativeStats(eeg_shape)
        embeddings_stats = {model_name: IterativeStats((ALIGN_ENCODER_DIM[model_name])) for model_name in config.model_names}

        for batch in tqdm.tqdm(dataloader, desc=f"Processing {split} split"):
            eeg = get_from_batch("eeg_data", batch, Tensor)
            eeg_stats.update(eeg)
            for model_name in config.model_names:
                emb = get_from_batch(model_name, batch, Tensor)
                embeddings_stats[model_name].update(emb)

        stat_dir = config.stat_path / "datasets" / data.config.dataset / split
        stat_dir.mkdir(parents=True, exist_ok=True)

        eeg_stats.save_to_path(stat_dir / "eeg")

        for model_name in config.model_names:
            embeddings_stats[model_name].save_to_path(stat_dir / model_name)

        logging.info(f"Saved stats for {split} split to {stat_dir}")
    logging.info(f"Finished processing data information")


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="generate_stats",
    version_base=None,
)
def main(cfg: DictConfig):
    setup()

    config = GetDataInformationConfig.from_hydra_config(cfg)
    logging.info("Starting data information generation")
    for key, value in config.model_dump(mode="json").items():
        logging.info(f"{key}: {value}")

    get_data_information(config)


if __name__ == "__main__":
    main()
