import logging
from typing import Any, Dict, cast
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from pathlib import Path
import sys
from brain_image.configs import GlobalConfig
from brain_image.utils import update_config_with_nested_key
from brain_image.trainer import NICETrainer

# Add project root to sys.path for direct imports
if str(GlobalConfig.WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(GlobalConfig.WORKSPACE_DIR))

from scripts.train_nice import train_nice


train_configs = {
    "trainer": None,
    "model": None,
    "dataset": None,
}


def create_sweep_config(cfg: DictConfig) -> Dict[str, Any]:
    sweep_cfg = cfg["sweep"]
    sweep_type = sweep_cfg["type"]
    sweep_params = sweep_cfg["parameters"][sweep_type]
    sweep_config = {
        "name": f"nice-{sweep_type}-sweep",
        "method": sweep_type,
        "metric": {
            "name": sweep_cfg["metric"]["name"],
            "goal": sweep_cfg["metric"]["goal"],
        },
        "parameters": OmegaConf.to_container(sweep_params, resolve=True),
        "early_terminate": {
            "type": sweep_cfg["early_terminate"]["type"],
            "min_iter": sweep_cfg["early_terminate"]["min_iter"],
        },
    }
    return sweep_config


def train_with_sweep_config():
    import time
    import random

    delay = random.uniform(0.5, 1.5)
    logging.info(f"Waiting {delay:.2f}s before starting...")
    time.sleep(delay)
    run = wandb.init(mode="online")

    sweep_config = wandb.config
    logging.info(f"Starting training run with sweep config: {dict(sweep_config)}")

    new_train_configs = {**train_configs}

    # Update training configs with sweep config
    # Sweep config is a flat dictionary, so we need to update the training configs with the sweep config
    # We need to handle nested dot notation in the keys
    for key, value in sweep_config.items():
        new_train_configs = update_config_with_nested_key(key, value, new_train_configs)

    if new_train_configs["trainer"] is None:
        raise ValueError("trainer_config is not set")
    if new_train_configs["model"] is None:
        raise ValueError("model_config is not set")
    if new_train_configs["dataset"] is None:
        raise ValueError("dataset_config is not set")

    logging.info(f"New train configs:")
    for key, value in new_train_configs.items():
        logging.info(f"  {key}: {value}")

    try:
        # Create trainer with the properly configured components
        trainer = NICETrainer(
            config=new_train_configs["trainer"],
            model_config=new_train_configs["model"],
            dataset_config=new_train_configs["dataset"],
        )

        model, test_metrics = train_nice(trainer)
        logging.info("✅ Training completed successfully")
        logging.info(f"Test metrics: {test_metrics}")
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise
    finally:
        run.finish()


def run_sweep(cfg: DictConfig):
    global train_configs
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg_dict is None:
        raise ValueError("cfg_dict is None")

    train_configs["trainer"] = cfg_dict["trainer"]
    train_configs["model"] = cfg_dict["model"]
    train_configs["dataset"] = cfg_dict["dataset"]

    sweep_cfg = cfg["sweep"]
    sweep_type = sweep_cfg["type"]
    sweep_count = sweep_cfg["count"]
    sweep_project = sweep_cfg["project"]
    sweep_entity = sweep_cfg["entity"]

    logging.info(f"Starting {sweep_type} sweep with {sweep_count} runs")
    logging.info(f"Project: {sweep_project}")
    logging.info(f"Entity: {sweep_entity}")
    wandb.init(project=sweep_project, entity=sweep_entity)

    sweep_config = create_sweep_config(cfg)
    logging.info(f"Sweep configuration: {sweep_config}")

    sweep_id = wandb.sweep(sweep_config, project=sweep_project, entity=sweep_entity)
    logging.info(f"✅ Created sweep with ID: {sweep_id}")
    logging.info(
        f"🌐 Sweep URL: https://wandb.ai/{sweep_entity or 'your-username'}/{sweep_project}/sweeps/{sweep_id}"
    )
    logging.info(f"🚀 Starting sweep agent for {sweep_count} runs...")
    wandb.agent(sweep_id, function=train_with_sweep_config, count=sweep_count)

    logging.info("🎉 Sweep completed!")
    return sweep_id


@hydra.main(
    config_path=str(GlobalConfig.CONFIGS_DIR),
    config_name="sweep",
    version_base=None,
)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Sweeping with config:")
    for key, value in cfg.items():
        logging.info(f"  {key}: {value}")

    torch.set_float32_matmul_precision("high")
    logging.info("🚀 Starting online-mode sweep")
    sweep_id = run_sweep(cfg)
    sweep_cfg = cfg["sweep"]
    logging.info(f"🎯 Sweep completed! Sweep ID: {sweep_id}")
    logging.info(
        f"📊 View results at: https://wandb.ai/{sweep_cfg['entity'] or 'your-username'}/{sweep_cfg['project']}/sweeps/{sweep_id}"
    )
    wandb.finish()


if __name__ == "__main__":
    main()
