#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <experiment_name> <param_path> <config_name> <train_script> <test_script> [cli_args...]" >&2
    exit 1
fi

experiment_name="$1"
param_path="$2"
config_name="$3"
train_script="$4"
test_script="$5"
shift 5

if [[ ! -f "$param_path" ]]; then
    echo "Error: parameter file not found: $param_path" >&2
    exit 1
fi

param_count="$(python scripts/slurm/param_parser.py "$param_path" --size)"
if [[ "$param_count" -lt 1 ]]; then
    echo "Error: parameter file contains no combinations: $param_path" >&2
    exit 1
fi

echo "Running $param_count local sweep tasks"
for task_id in $(seq 0 "$((param_count - 1))"); do
    scripts/evaluation/run_experiment_task_local.sh \
        "$experiment_name" "$task_id" "$param_path" \
        "$config_name" "$train_script" "$test_script" "$@"
done

echo "Aggregating sweep results"
python scripts/evaluation/aggregate_metrics.py \
    --experiment_dir "experiments/$experiment_name" \
    --metrics_file_pattern '*test_metrics.csv' \
    --hparams \
        model.align_img_encoder \
        model.eeg_encoder.eeg_encoder \
        model.seed \
        dataset.subs

echo "Sweep complete: experiments/$experiment_name/aggregated_metrics.csv"
