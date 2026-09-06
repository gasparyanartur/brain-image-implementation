import argparse
import csv
import logging
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from omegaconf import DictConfig

from hydra import compose, initialize_config_dir
from brain_image.configs import GlobalConfig
from brain_image.model.low_level import LowLevelModule
from brain_image.eval import find_checkpoint_in_run
from brain_image.utils import flatten_configs, setup
from scripts.training.train_low_level import TrainLowLevelConfig


class EvaluationConfig(TrainLowLevelConfig):
    output_dir: Path | None = None


def _find_run_path(path: Path) -> Path:
    if (path / "checkpoints").exists():
        return path
    version_dirs = sorted(path.glob("version_*"), reverse=True)
    if not version_dirs:
        version_dirs = sorted(path.glob("*/version_*"), reverse=True)
    for version_dir in version_dirs:
        if list((version_dir / "checkpoints").glob("*.ckpt")):
            return version_dir
    return path


def main(cfg: DictConfig):
    setup()
    args = EvaluationConfig.from_hydra_config(cfg)
    run_path = _find_run_path(Path(cfg.run_path)) if "run_path" in cfg else None
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if checkpoint_path is None and run_path is not None:
        checkpoint_dir = run_path / "checkpoints"
        try:
            checkpoint_path = find_checkpoint_in_run(
                checkpoint_dir, "min", args.trainer.checkpoint_monitor
            )
        except IndexError:
            checkpoint_path = checkpoint_dir / "last.ckpt"
    if checkpoint_path is None:
        raise ValueError("Provide checkpoint_path or run_path")

    args.checkpoint_path = str(checkpoint_path)
    logging.info("Evaluating checkpoint %s", checkpoint_path)
    model = LowLevelModule(args.model, args.dataset, compile=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    evaluator = pl.Trainer(
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    results = evaluator.test(model=model, verbose=False)
    metrics = results[0] if results else {}

    output_dir = args.output_dir or checkpoint_path.parent.parent / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "value"))
        writer.writerows((name, value) for name, value in metrics.items())
    with (output_dir / "evaluation_config.yaml").open("w") as file:
        yaml.safe_dump(
            {"checkpoint_path": str(checkpoint_path), **args.model_dump(mode="json")},
            file,
            sort_keys=False,
        )

    logging.info("Saved metrics to %s", output_dir / "test_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--limit_test_size", type=float)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--config_name", default="train_low_level")
    cli_args = parser.parse_args()
    overrides = [
        f"+run_path={cli_args.run_path}",
        "trainer.compile_model=false",
        "trainer.wandb.enabled=false",
        f"dataset.batch_size={cli_args.batch_size}",
        f"dataset.val_batch_size={cli_args.batch_size}",
        f"dataset.test_batch_size={cli_args.batch_size}",
        f"model.eval_batch_size={cli_args.batch_size}",
    ]
    if cli_args.checkpoint_path:
        overrides.append(f"checkpoint_path={cli_args.checkpoint_path}")
    if cli_args.output_dir:
        overrides.append(f"+output_dir={cli_args.output_dir}")
    if cli_args.limit_test_size is not None:
        overrides.append(f"dataset.limit_test_size={cli_args.limit_test_size}")
    with initialize_config_dir(config_dir=str(GlobalConfig.CONFIGS_DIR), version_base=None):
        main(compose(config_name=cli_args.config_name, overrides=overrides))
