import logging

import hydra
from omegaconf import DictConfig
from brain_image.configs import BaseConfig
from brain_image.data.dataset.union import EEGDatasetConfigType
from brain_image.trainer import EEGAlignTrainer, EEGAlignTrainerConfig
from brain_image.model.eeg_alignment import EEGAlignmentConfig, EEGAlignmentModel
from brain_image.data.dataset.eeg_dataset import EEGDatasetConfig

from pathlib import Path

from brain_image.configs import GlobalConfig
from brain_image.utils import flatten_configs, get_dtype, setup


class TrainEEGConfig(BaseConfig):
    dataset: EEGDatasetConfigType
    model: EEGAlignmentConfig 
    trainer: EEGAlignTrainerConfig 

    checkpoint_path: str | None = None
    resume_training: bool = False


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="train_eeg",
    version_base=None,
)
def main(cfg: DictConfig):
    setup()

    try:
        config = TrainEEGConfig.from_hydra_config(cfg)
    except BaseException as e:
        logging.error("Failed to parse config:")
        for key, value in flatten_configs(cfg).items():
            logging.error(f"  {key}: {value}")
        raise e

    logging.info(f"Training with config:")
    for key, value in flatten_configs(config).items():
        logging.info(f"  {key}: {value}")

    model = EEGAlignmentModel(
        config=config.model,
        dataset_config=config.dataset,
        compile=config.trainer.compile_model,
        init_weights=config.trainer.init_weights,
        dtype=get_dtype(config.trainer.dtype),
        cache_dir=config.trainer.cache_dir,
    )

    trainer = EEGAlignTrainer(
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
