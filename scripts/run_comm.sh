#!/usr/bin/env bash
# Train and evaluate one local CoMM experiment.
#
# Usage:
#   scripts/run_comm.sh [Hydra overrides...]

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

experiment_root="${COMM_OUTPUT_ROOT:-experiments/comm}"
run_group="${experiment_root}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_group"

echo "Training CoMM with config train_comm"
echo "Output root: $run_group"

python scripts/training/train_comm.py \
    --config-name=train_comm \
    "trainer.log_dir=$run_group" \
    "trainer.wandb.enabled=false" \
    "$@"

run_dir="$(find "$run_group" -mindepth 2 -maxdepth 2 -type d -path '*/version_0' -printf '%h\n' | sort | tail -n 1)"
if [[ -z "$run_dir" ]]; then
    echo "Error: could not find a completed Lightning run under $run_group" >&2
    exit 1
fi

echo "Evaluating run: $run_dir"
python scripts/evaluation/test_comm.py \
    "$run_dir" \
    --checkpoint_selection max \
    --checkpoint_metric val/acc_eeg_to_img

echo "Experiment complete"
echo "Run directory: $run_dir"
echo "Training TensorBoard: $run_dir/version_0"
echo "Evaluation metrics: $run_dir/version_0/test/test_metrics.csv"
echo "Evaluation config: $run_dir/version_0/test/evaluation_config.yaml"
