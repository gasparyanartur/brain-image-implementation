import logging
from typing import cast

import hydra
from omegaconf import DictConfig
from brain_image.configs import BaseConfig
from brain_image.data.dataset.union import EEGDatasetConfigType
from brain_image.model.comm_alignment import CommAlignmentConfig, CommAlignmentModel
from brain_image.trainer import CommAlignTrainer, CommAlignTrainerConfig
from brain_image.data.dataset.eeg_dataset import EEGDatasetConfig

from pathlib import Path

from brain_image.configs import GlobalConfig
from brain_image.utils import flatten_configs, setup


class TrainCommConfig(BaseConfig):
    dataset: EEGDatasetConfigType
    model: CommAlignmentConfig
    trainer: CommAlignTrainerConfig = CommAlignTrainerConfig()

    checkpoint_path: str | None = None
    resume_training: bool = False
    cache_images: bool = True
    preload_images: bool = True


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="train_comm",
    version_base=None,
)
def main(cfg: DictConfig):
    setup()

    try:
        config = cast(TrainCommConfig, TrainCommConfig.from_hydra_config(cfg))
    except BaseException as e:
        logging.error("Failed to parse config:")
        for key, value in flatten_configs(cfg).items():
            logging.error(f"  {key}: {value}")
        raise e

    logging.info(f"Training with config:")
    for key, value in flatten_configs(config).items():
        logging.info(f"  {key}: {value}")

    model = CommAlignmentModel(
        config=config.model,
        dataset_config=config.dataset,
        compile=config.trainer.compile_model,
        preload_images=config.preload_images,
        cache_images=config.cache_images
    )

    trainer = CommAlignTrainer(
        trainer_config=config.trainer,
        model=model   
    )

    if (cp_path := config.checkpoint_path) and (checkpoint_path := Path(cp_path)).exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

    trainer.train()
    test_metrics = trainer.test()

    logging.info(f"Finished training with test metrics:")
    for key, value in test_metrics.items():
        logging.info(f"  {key}: {value}")


if __name__ == "__main__":
     main()
