import datetime
import argparse
import logging
from pathlib import Path

from typing import Literal
from pydantic import BaseModel
import json

import torch

from brain_image.eval import find_checkpoint_in_run
from brain_image.metrics import METRIC_LOOKUP, MetricType
from brain_image.utils import flatten_configs, setup, setup_logging
from brain_image.model.comm_alignment import CommAlignmentModel
import os

from torchvision.utils import save_image


class Args(BaseModel):
    run_path: Path
    checkpoint_path: Path | None = None
    hyperparameters_path: Path | None = None
    checkpoint_selection: Literal["last", "max", "min"] = "min"
    checkpoint_metric: str = "val-loss"
    output_dir: Path | None = None
    disable_cache: bool = False


def main(args: Args):
    setup()

    logging.info(f"Running with args:")
    for key, value in flatten_configs(args).items():
        logging.info(f"  {key}: {value}")

    if version_dirs := list(args.run_path.glob("version_*")):
        for version_dir in reversed(sorted(version_dirs)):
            if version_dir.glob("*.ckpt"):
                args.run_path = version_dir
                break
        else:
            raise ValueError(
                f"No checkpoints found in any version of {list(v.name for v in version_dirs)}"
            )

    if args.checkpoint_path is None:
        logging.info(
            f"Checkpoint path not specified, selecting checkpoint from {args.run_path}"
        )
        cp_dir = args.run_path / "checkpoints"
        args.checkpoint_path = find_checkpoint_in_run(
            cp_dir, args.checkpoint_selection, args.checkpoint_metric
        )
        logging.info(f"Selected checkpoint: {args.checkpoint_path}")

    logging.info(
        f"Loading model from {args.checkpoint_path} with hyperparameters from {args.hyperparameters_path}..."
    )

    model = CommAlignmentModel.load_from_checkpoint(
        args.checkpoint_path,
        hparams_file=args.hyperparameters_path,
        cache_images=not args.disable_cache,
        preload_images=not args.disable_cache,
        strict=False
    )
    model.eval()
    logging.info(f"Finished loading model.")

    logging.info(f"Running full test...")
    loader = model.data_module.get_dataloader("test")
    metrics = model.run_full_test(loader)
    logging.info(f"Finished running full test.")

    logging.info(f"Metrics:")
    for k, v in metrics.items():
        logging.info(
            f"  {k}: {v}"
        )

    name = args.run_path.name
    output_dir = (
        (args.output_dir / name)
        if (args.output_dir is not None)
        else (args.run_path / "test")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(args.model_dump_json(indent=4), f)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path, help="Path to the run")
    parser.add_argument(
        "--checkpoint_path",
        "-c",
        type=Path,
        help="Path to the checkpoint, overrides checkpoints found in the run path",
    )
    parser.add_argument(
        "--hyperparameters_path", "-hp", type=Path, help="Path to the hyperparameters"
    )
    parser.add_argument(
        "--checkpoint_selection",
        choices=["last", "max", "min"],
        default="min",
        help="How to select the checkpoint",
    )
    parser.add_argument(
        "--checkpoint_metric",
        default="val-loss",
        help="Metric used to find best checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        help="Experiment directory. if None, results are written to run_path/outputs",
        default=None,
    )
    parser.add_argument(
        "--disable_cache", action="store_true", help="Disable Cache images in memory"
    )

    args = parser.parse_args()
    main(Args(**vars(args)))
