import logging

import hydra
from omegaconf import DictConfig
from brain_image.configs import BaseConfig
from brain_image.model.low_level import LowLevelConfig, LowLevelModule
from brain_image.data.dataset.union import EEGDatasetConfigType

from pathlib import Path

from brain_image.configs import GlobalConfig
from brain_image.trainer import Trainer, TrainerConfig
from brain_image.utils import flatten_configs, setup


class TrainLowLevelConfig(BaseConfig):
    dataset: EEGDatasetConfigType
    model: LowLevelConfig 
    trainer: TrainerConfig 

    checkpoint_path: str | None = None
    resume_training: bool = False


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="train_low_level",
    version_base=None,
)
def main(cfg: DictConfig):
    setup()

    try:
        config = TrainLowLevelConfig.from_hydra_config(cfg)
    except BaseException as e:
        logging.error("Failed to parse config:")
        for key, value in flatten_configs(cfg).items():
            logging.error(f"  {key}: {value}")
        raise e

    logging.info(f"Training with config:")
    for key, value in flatten_configs(config).items():
        logging.info(f"  {key}: {value}")

    model = LowLevelModule(
        config=config.model,
        dataset_config=config.dataset,
        compile=config.trainer.compile_model,
    )

    trainer = Trainer(
        config=config.trainer,
        model=model   
    )

    if (cp_path := config.checkpoint_path) and (checkpoint_path := Path(cp_path)).exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

    trainer.train()
    logging.info("Finished training. Run scripts/evaluation/test_low_level.py on the selected checkpoint.")


if __name__ == "__main__":
    main()
