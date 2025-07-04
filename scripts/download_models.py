import logging
from pathlib import Path
import dreamsim
import hydra
from omegaconf import DictConfig

from brain_image.configs import BaseConfig
from brain_image.model import load_image_encoder


class DownloadModelsConfig(BaseConfig):
    models: list[str] = ["synclr", "aligned_synclr"]
    model_path: Path = Path("models")


@hydra.main(config_path="../configs", config_name="download_models", version_base=None)
def main(cfg: DictConfig):
    config = DownloadModelsConfig.from_hydra_config(cfg)
    logging.basicConfig(level=logging.INFO)

    dreamsim_types = []
    for model in config.models:
        if model in {"synclr", "aligned_synclr"}:
            dreamsim_types.append("synclr_vitb16")
        else:
            raise ValueError(f"Unknown model type: {model}")

    logging.info(
        f"Downloading models: {dreamsim_types} to {config.model_path.absolute()}"
    )
    for dreamsim_type in dreamsim_types:
        try:
            logging.info(f"Downloading {dreamsim_type} model...")
            dreamsim.model.download_weights(
                dreamsim_type=dreamsim_type,
                cache_dir=config.model_path,
            )
        except Exception as e:
            logging.error(f"Error downloading {dreamsim_type} model: {e}")
        else:
            logging.info(f"Model {dreamsim_type} downloaded successfully.")
    logging.info("Finished downloading models.")

    logging.info(f"Validating models: {dreamsim_types}")
    for model in config.models:
        try:
            load_image_encoder(model_name=model, models_path=config.model_path)
        except Exception as e:
            logging.error(f"Error loading {model} model: {e}")
            return

    logging.info("All models downloaded and validated successfully.")


if __name__ == "__main__":
    main()
