import argparse
import pandas as pd
import logging
from pathlib import Path
import json
import yaml

from brain_image.utils import setup


def get_single_file(dir: Path, pattern: str) -> Path | None:
    paths = list(dir.rglob(pattern))

    num_results = len(paths)
    if num_results == 0:
        return None

    if num_results > 1:
        raise ValueError(f"Expected to find one result matching pattern {pattern} in dir {dir} - Found {num_results}: {tuple(paths)}")

    return paths[0]


def gather_metrics(experiment_dir: Path, selected_hparams: list[str], metrics_file_pattern: str = "*test/test_metrics.json") -> pd.DataFrame:
    all_metrics = []

    for exp_dir in experiment_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        metrics_path = get_single_file(exp_dir, metrics_file_pattern)
        if metrics_path is None:
            logging.warning(f"Could not find any paths in dir {exp_dir} matching pattern {metrics_file_pattern}")
            continue

        logging.info(f"Loading metrics from {metrics_path}")

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["run"] = exp_dir.name

        if selected_hparams:
            hparams_path = get_single_file(exp_dir, "*hparams.yaml")
            if hparams_path is None:
                logging.warning(f"Could not find hparam file in {exp_dir}")
                continue

            with open(hparams_path) as f:
                hparams = yaml.safe_load(f)

            for hparam_key in selected_hparams:
                hparam_parts = hparam_key.split(".")
                curr_hparam = hparams
                for part in hparam_parts:
                    curr_hparam = curr_hparam.get(part, None)
                    if curr_hparam is None:
                        break

                metrics[hparam_key] = curr_hparam

        all_metrics.append(metrics)

    return pd.DataFrame.from_records(all_metrics)


def main():
    setup()

    parser = argparse.ArgumentParser(description="Aggregate metrics from experiment results.")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Directory containing experiment run subdirectories.')
    parser.add_argument('--hparams', type=str, nargs='*', default=[], help='Dotted hparam keys to include from hparams.yaml (e.g. model.lr).')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the aggregated CSV. Defaults to experiment_dir.')
    parser.add_argument('--metrics_file_pattern', type=str, default="*test/test_metrics.json", help='Glob pattern to find metrics files within each run directory.')
    parser.add_argument("--output_name", type=str, default="aggregated_metrics.csv", help="Name of the output CSV file.")

    args = parser.parse_args()
    logging.info(f"Aggregating metrics from experiment directory: {args.experiment_dir}")

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = gather_metrics(experiment_dir, args.hparams, args.metrics_file_pattern)

    output_path = output_dir / args.output_name
    metrics.to_csv(output_path, index=False)
    logging.info(f"Aggregated metrics saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
