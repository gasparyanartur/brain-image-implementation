#!/usr/bin/env bash
# Train and evaluate one local EEG alignment experiment.
#
# Usage:
#   scripts/run_eeg_alignment.sh [Hydra overrides...]
#
# Example:
#   scripts/run_eeg_alignment.sh model.max_epochs=1 trainer.wandb.enabled=false

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

experiment_root="${EEG_ALIGNMENT_OUTPUT_ROOT:-experiments/eeg_alignment}"
run_group="${experiment_root}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_group"

echo "Training EEG alignment with config train_eeg_align"
echo "Output root: $run_group"

python scripts/training/train_eeg.py \
    --config-name=train_eeg_align \
    "trainer.log_dir=$run_group" \
    "trainer.wandb.enabled=false" \
    "$@"

run_dir="$(find "$run_group" -mindepth 2 -maxdepth 2 -type d -path '*/version_0' -printf '%h\n' | sort | tail -n 1)"
if [[ -z "$run_dir" ]]; then
    echo "Error: could not find a completed Lightning run under $run_group" >&2
    exit 1
fi

echo "Evaluating run: $run_dir"
python scripts/evaluation/test_eeg.py "$run_dir"

echo "Experiment complete"
echo "Run directory: $run_dir"
echo "Training TensorBoard: $run_dir/version_0"
echo "Evaluation metrics: $run_dir/version_0/test/test_metrics.csv"
echo "Evaluation config: $run_dir/version_0/test/evaluation_config.yaml"
