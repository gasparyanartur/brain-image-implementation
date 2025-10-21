import datetime
import re
import argparse 
import logging
from pathlib import Path

from typing import Literal
from pydantic import BaseModel
import json

import torch

from brain_image.metrics import METRIC_LOOKUP, MetricType
from brain_image.model.eeg_alignment import EEGAlignmentModel
from brain_image.utils import flatten_configs, setup_logging
import os

from torchvision.utils import save_image


class Args(BaseModel):
    run_path: Path | None = None
    checkpoint_path: Path | None = None
    hyperparameters_path: Path | None = None
    checkpoint_selection: Literal["last", "max", "min"] = "min"
    checkpoint_metric: str = "val-loss"
    output_dir: Path = Path("outputs/experiments")
    metrics: list[str] = ['pixcorr', 'ssim', 'alex2', 'alex5', 'inceptionv3', 'clip', 'efficientnet', 'swav']
    recon_idxs: list[int] | None = None


def _find_cp_to_use(checkpoint_dir: Path, checkpoint_selection: Literal["last", "max", "min"], checkpoint_metric: str) -> Path:
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")

    cp_candidates = list(checkpoint_dir.glob("*.ckpt"))
    if len(cp_candidates) == 0:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    name_pattern = re.compile(rf".+epoch_(\d+)-{checkpoint_metric}_([0-9]*\.?[0-9]+)\.ckpt") 

    parsed_infos = []
    logging.info(f"Found {len(cp_candidates)} checkpoints in {checkpoint_dir}")
    for cp in cp_candidates:
        if cp.name == "last.ckpt":
            if checkpoint_selection == "last":
                return cp
            else:
                continue
        
        logging.debug(f"Parsing checkpoint {cp.name}")
        parsed_info = name_pattern.match(cp.name)
        if parsed_info is None:
            logging.warning(f"Checkpoint {cp} does not match the pattern {name_pattern} - Ignoring this one as candidate")
            continue
        
        epoch = int(parsed_info.group(1))
        value = float(parsed_info.group(2))
        parsed_infos.append({"epoch": epoch, "value": value, "path": cp})

    for info in parsed_infos:
        logging.debug(f"Checkpoint {info['path']} has epoch {info['epoch']} and value {info['value']}")

    match checkpoint_selection:
        case "last":
            sort_strategy = lambda info: -info["epoch"]
        case "max":
            sort_strategy = lambda info: -info["value"]
        case "min":
            sort_strategy = lambda info: info["value"]
        
    parsed_infos.sort(key=sort_strategy)
    return parsed_infos[0]["path"]


def main(args: Args):
    setup_logging()

    logging.info(f"Running with args:") 
    for key, value in flatten_configs(args).items():
        logging.info(f"  {key}: {value}")

    if args.checkpoint_path is None:
        if args.run_path is None:
            raise ValueError(f"If checkpoint_path is not specified, run_path must be specified")

        logging.info(f"Checkpoint path not specified, selecting checkpoint from {args.run_path}")
        args.checkpoint_path = _find_cp_to_use(args.run_path / "checkpoints", args.checkpoint_selection, args.checkpoint_metric)
        logging.info(f"Selected checkpoint: {args.checkpoint_path}")


    logging.info(f"Loading model from {args.checkpoint_path} with hyperparameters from {args.hyperparameters_path}...")
    model = EEGAlignmentModel.load_from_checkpoint(args.checkpoint_path, hparams_file=args.hyperparameters_path)
    model.eval()
    logging.info(f"Finished loading model.")
    
    logging.info(f"Running full test...")
    metrics, imgs, outputs = model.run_full_test(metrics=args.metrics, recon_idxs=args.recon_idxs if args.recon_idxs else None)
    logging.info(f"Finished running full test.")

    logging.info(f"Metrics:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value}")

    metrics = {name.split("/")[-1]: value.item() for name, value in metrics.items()}   # Remove the prefix
    imgs = {name.split("/")[-1]: value for name, value in imgs.items()}   # Remove the prefix
    outputs = {name.split("/")[-1]: value for name, value in outputs.items()}   # Remove the prefix

    name = model.get_name(timestamp=False)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    name = f"{name}_{timestamp}"
    output_path = args.output_dir / name
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(output_path / "config.json", "w") as f:
        json.dumps(
            args.model_dump_json(indent=4)
        )

    reconstructions = imgs["reconstruction"] 
    ground_truths = imgs["ground_truth"]
    idxs = outputs["idx"]
    img_paths = outputs["img_path"]
    img_dir = Path(output_path / "imgs")
    img_dir.mkdir(parents=True, exist_ok=True)
    
    for reconstruction, ground_truth, idx, img_path in zip(reconstructions, ground_truths, idxs, img_paths):
        save_image(reconstruction, img_dir / f"{idx}_recon.jpg")
        save_image(ground_truth, img_dir / f"{idx}_gt.jpg")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", "-r", type=Path, help="Path to the run")
    parser.add_argument("--checkpoint_path", "-c", type=Path, help="Path to the checkpoint")
    parser.add_argument("--hyperparameters_path", "-hp", type=Path, help="Path to the hyperparameters")
    parser.add_argument("--checkpoint_selection", choices=["last", "max", "min"], default="min", help="How to select the checkpoint")
    parser.add_argument("--checkpoint_metric", default="val-loss", help="Metric used to find best checkpoint")
    parser.add_argument("--output_dir", "-o", type=Path, help="Output directory", default=Path("outputs/experiments"))
    parser.add_argument("--metrics", "-m", type=str, nargs="+", default=list(METRIC_LOOKUP.keys()), choices=list(METRIC_LOOKUP.keys()), help="Metrics to compute")
    parser.add_argument("--recon_idxs", "-i", type=int, nargs="*", default=None)


    args = parser.parse_args()
    main(Args(**vars(args)))


