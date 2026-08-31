import argparse
import csv
import logging
from pathlib import Path

from typing import Literal
from pydantic import BaseModel

import torch
import yaml

from brain_image.eval import find_checkpoint_in_run
from brain_image.metrics import METRIC_LOOKUP
from brain_image.model.eeg_alignment import EEGAlignmentModel
from brain_image.model.model import dump_test_output
from brain_image.utils import flatten_configs, setup_logging


class Args(BaseModel):
    run_path: Path
    checkpoint_path: Path | None = None
    hyperparameters_path: Path | None = None
    checkpoint_selection: Literal["last", "max", "min"] = "min"
    checkpoint_metric: str = "val/align/top1"
    output_dir: Path | None = None
    metrics: list[str] = ['pixcorr', 'ssim', 'alex2', 'alex5', 'inceptionv3', 'clip', 'efficientnet', 'swav']
    recon_idxs: list[int] | None = None


def main(args: Args):
    setup_logging()

    logging.info(f"Running with args:") 
    for key, value in flatten_configs(args).items():
        logging.info(f"  {key}: {value}")

    if (version_dirs := list(args.run_path.glob("version_*"))):
        for version_dir in reversed(sorted(version_dirs)):
            if version_dir.glob("*.ckpt"):
                args.run_path = version_dir
                break
        else:
            raise ValueError(f"No checkpoints found in any version of {list(v.name for v in version_dirs)}")


    if args.checkpoint_path is None:
        logging.info(f"Checkpoint path not specified, selecting checkpoint from {args.run_path}")
        cp_dir = args.run_path / "checkpoints"
        args.checkpoint_path = find_checkpoint_in_run(cp_dir, args.checkpoint_selection, args.checkpoint_metric)
        logging.info(f"Selected checkpoint: {args.checkpoint_path}")

    logging.info(f"Loading model from {args.checkpoint_path} with hyperparameters from {args.hyperparameters_path}...")
    model = EEGAlignmentModel.load_from_checkpoint(args.checkpoint_path, hparams_file=args.hyperparameters_path)
    model.eval()
    logging.info(f"Finished loading model.")

    # Force full image-reconstruction evaluation at test time, regardless of
    # whether training skipped it to save epoch-level validation cost.
    model.config.skip_reconstruction = False
    if args.metrics:
        model.config.test_recon_metrics = list(args.metrics)  # type: ignore[assignment]
    if args.recon_idxs is not None:
        model.config.highlighted_test_recons = list(args.recon_idxs)
    logging.info(f"Test-time recon enabled. Metrics: {model.config.test_recon_metrics}")

    logging.info(f"Running full test...")
    metrics, imgs = model.run_full_validation(split="test")
    logging.info(f"Finished running full test.")

    logging.info(f"Metrics:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value}")


    name = args.run_path.name
    output_dir = (args.output_dir / name) if (args.output_dir is not None) else (args.run_path / "test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "test_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("metric", "value"))
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.detach().cpu().item()
            writer.writerow((metric_name, metric_value))

    with open(output_dir / "evaluation_config.yaml", "w") as f:
        yaml.safe_dump(args.model_dump(mode="json"), f, sort_keys=False)

    dump_test_output(output_dir, metrics, imgs, metrics_file_name=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path, help="Path to the run")
    parser.add_argument("--checkpoint_path", "-c", type=Path, help="Path to the checkpoint, overrides checkpoints found in the run path")
    parser.add_argument("--hyperparameters_path", "-hp", type=Path, help="Path to the hyperparameters")
    parser.add_argument("--checkpoint_selection", choices=["last", "max", "min"], default="min", help="How to select the checkpoint")
    parser.add_argument("--checkpoint_metric", default="val/align/top1", help="Metric used to find best checkpoint")
    parser.add_argument("--output_dir", "-o", type=Path, help="Experiment directory. if None, results are written to run_path/outputs", default=None)
    parser.add_argument("--metrics", "-m", type=str, nargs="+", default=list(METRIC_LOOKUP.keys()), choices=list(METRIC_LOOKUP.keys()), help="Metrics to compute")
    parser.add_argument("--recon_idxs", "-i", type=int, nargs="*", default=None)


    args = parser.parse_args()
    main(Args(**vars(args)))
