#!/bin/bash
# Run a single parameter configuration locally (no SLURM, no Singularity).
#
# Usage:
#   ./scripts/evaluation/run_experiment_task_local.sh <experiment_name> <task_id> <param_path> <config_name> <train_script> <test_script> [cli_args...]
#
# Arguments:
#   experiment_name Name for the experiment (results go to experiments/<name>/<timestamp>_task<id>).
#   task_id         0-based index into the parameter combinations defined in param_path.
#                   The all-task launcher handles aggregation after the sweep.
#   param_path      Path to a param_parser JSON file defining the sweep.
#   config_name     Hydra config name passed to the training script.
#   train_script    Python training script (e.g. scripts/training/train_eeg.py).
#   test_script     Python evaluation script (e.g. scripts/evaluation/test_eeg.py).
#   cli_args        Extra Hydra overrides forwarded verbatim to the training script.
#
# Examples:
#   # Run the 3rd (0-based) encoder configuration from the text_vs_img sweep:
#   ./scripts/evaluation/run_experiment_task_local.sh text_vs_img 2 scripts/params/text_vs_img_encoders.json \
#       train_eeg_align_text scripts/training/train_eeg.py scripts/evaluation/test_eeg.py
#
#
#   # List all parameter combinations to find the right index:
#   python scripts/slurm/param_parser.py scripts/params/text_vs_img_encoders.json -s
#   for i in $(seq 0 5); do
#       echo "$i: $(python scripts/slurm/param_parser.py scripts/params/text_vs_img_encoders.json -i $i)"
#   done

set -euo pipefail

_root="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -f "$_root/.env" ]]; then
    set -a; source "$_root/.env"; set +a
fi

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <experiment_name> <task_id> <param_path> <config_name> <train_script> <test_script> [cli_args...]"
    echo ""
    echo "  experiment_name  Name for the experiment (results go to experiments/<name>/...)"
    echo "  task_id          0-based index of the parameter combination"
    echo "  param_path       Path to the param_parser JSON file"
    echo "  config_name      Hydra config name (not needed for AGG)"
    echo "  train_script     Python training script (not needed for AGG)"
    echo "  test_script      Python evaluation script (not needed for AGG)"
    exit 1
fi

experiment_name="$1"
task_id="$2"
param_path="$3"
config_name="$4"
train_script="$5"
test_script="$6"
shift 6
extra_cli_args="${*:-}"

if [[ ! -f "$param_path" ]]; then
    echo "Error: param file not found: $param_path"
    exit 1
fi

param_count=$(python scripts/slurm/param_parser.py "$param_path" -s)
echo "Total parameter combinations: $param_count"

if [[ "$task_id" -ge "$param_count" ]]; then
    echo "Error: task_id=$task_id is out of range (0..$((param_count - 1)))"
    exit 1
fi

# ── Resolve parameter combination ─────────────────────────────────────────────

task_args=$(python scripts/slurm/param_parser.py "$param_path" -i "$task_id")

experiment_dir="experiments/${experiment_name}/$(date +%Y%m%d_%H%M%S)_task${task_id}"
mkdir -p "$experiment_dir"

echo "=== Local Experiment Launch ==="
echo "  Task ID:      $task_id / $((param_count - 1))"
echo "  Params:       $task_args"
echo "  Config:       $config_name"
echo "  Train script: $train_script"
echo "  Test script:  $test_script"
echo "  Experiment:   $experiment_dir"
[[ -n "$extra_cli_args" ]] && echo "  Extra args:   $extra_cli_args"
echo ""

# ── Train ─────────────────────────────────────────────────────────────────────

echo "--- Training ---"
# shellcheck disable=SC2086
python "$train_script" \
    --config-name="$config_name" \
    trainer.log_dir="$experiment_dir" \
    $task_args \
    ${extra_cli_args}

# ── Find run directory ────────────────────────────────────────────────────────

echo ""
echo "--- Locating run directory ---"
run_dir=$(ls -td "${experiment_dir}/"*/ 2>/dev/null | head -1)
run_dir="${run_dir%/}"

if [[ -z "${run_dir:-}" ]]; then
    echo "Error: could not find run directory in $experiment_dir — skipping test"
    exit 1
fi
echo "Run directory: $run_dir"

# ── Test ──────────────────────────────────────────────────────────────────────

echo ""
echo "--- Testing ---"
python "$test_script" "$run_dir"

echo ""
echo "Done. Results in: $run_dir"
