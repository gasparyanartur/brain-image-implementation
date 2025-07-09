import logging
from typing import Any, Dict, cast

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from pathlib import Path
from brain_image.configs import GlobalConfig


def create_sweep_config(cfg: DictConfig) -> Dict[str, Any]:
    """Create wandb sweep configuration from Hydra config."""
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
        "parameters": sweep_params,
        "early_terminate": {
            "type": sweep_cfg["early_terminate"]["type"],
            "min_iter": sweep_cfg["early_terminate"]["min_iter"],
        },
    }
    # Convert to regular Python dict to make it JSON serializable
    return cast(Dict[str, Any], OmegaConf.to_container(sweep_config, resolve=True))


def train_with_sweep_config():
    """Training function that will be called by wandb agent."""
    run = wandb.init()
    sweep_config = wandb.config
    logging.info(f"Starting training run with sweep config: {dict(sweep_config)}")
    try:
        overrides = [f"{key}={value}" for key, value in sweep_config.items()]
        logging.info(f"Hydra overrides: {overrides}")
        import os

        os.environ["HYDRA_OVERRIDES"] = " ".join(overrides)
        from scripts.train_nice import main as train_main

        train_main()
        logging.info("✅ Training completed successfully")
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise


def run_sweep(cfg: DictConfig):
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
    torch.set_float32_matmul_precision("high")
    sweep_id = run_sweep(cfg)
    sweep_cfg = cfg["sweep"]
    logging.info(f"🎯 Sweep completed! Sweep ID: {sweep_id}")
    logging.info(
        f"📊 View results at: https://wandb.ai/{sweep_cfg['entity'] or 'your-username'}/{sweep_cfg['project']}/sweeps/{sweep_id}"
    )


if __name__ == "__main__":
    main()
