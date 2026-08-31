import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from brain_image.configs import GlobalConfig
from brain_image.configs import BaseConfig
from brain_image.data.dataset.union import EEGDatasetConfigType
from brain_image.model.eeg_alignment import EEGAlignmentConfig, EEGAlignmentModel
from brain_image.trainer import Trainer, TrainerConfig
from brain_image.utils import flatten_configs, get_dtype, setup


class TrainEEGPriorConfig(BaseConfig):
    dataset: EEGDatasetConfigType
    model: EEGAlignmentConfig
    trainer: TrainerConfig

    checkpoint_path: str | None = None
    resume_training: bool = False


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="train_eeg_prior",
    version_base=None,
)
def main(cfg: DictConfig):
    setup()

    try:
        config = TrainEEGPriorConfig.from_hydra_config(cfg)
    except BaseException as error:
        logging.error("Failed to parse config:")
        for key, value in flatten_configs(cfg).items():
            logging.error(f"  {key}: {value}")
        raise error

    logging.info("Training EEG prior with config:")
    for key, value in flatten_configs(config).items():
        logging.info(f"  {key}: {value}")

    model = EEGAlignmentModel(
        config=config.model,
        dataset_config=config.dataset,
        compile=config.trainer.compile_model,
        dtype=get_dtype(config.trainer.dtype),
        cache_dir=config.trainer.cache_dir,
    )

    trainer = Trainer(config=config.trainer, model=model)

    if (checkpoint_path := config.checkpoint_path) and Path(checkpoint_path).exists():
        logging.info(f"Resuming full prior checkpoint from {checkpoint_path}")
        trainer.load_checkpoint(Path(checkpoint_path))

    trainer.train()
    logging.info("Finished prior training. Run the separate EEG evaluation script to compute reconstruction metrics.")


if __name__ == "__main__":
    main()
