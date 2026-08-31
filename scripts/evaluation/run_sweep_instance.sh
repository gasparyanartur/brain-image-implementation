#!/bin/bash
# SLURM array job body for a sweep. Loads the sweep params for this task from a
# param_parser JSON file, then delegates to run_experiment.sh.
#
# Submit via run_experiment_sweep_slurm.sh, or manually:
#   SBATCH_ARRAY=0-N SSUB_NO_SINGULARITY=1 \
#     ./scripts/slurm/ssub.sh <name> \
#       bash scripts/evaluation/run_sweep_instance.sh \
#         <param_path> <config_name> <train_script> <test_script> [--experiment_dir <dir>] [cli_args...]
#
# Args:
#   param_path      Path to a param_parser JSON file.
#   config_name     Hydra config name passed to the train script.
#   train_script    Python training script (e.g. scripts/training/train_eeg.py).
#   test_script     Python evaluation script (e.g. scripts/evaluation/test_eeg.py).
#
# Options:
#   --experiment_dir <dir>   Forwarded to run_experiment.sh (default: logs/experiments).
#
# cli_args: remaining args are forwarded verbatim as Hydra overrides.
#
# Env overrides:
#   TASK_ID   Override SLURM_ARRAY_TASK_ID (useful for local testing).

set -euo pipefail

echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Array Index:  ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Job Name:     ${SLURM_JOB_NAME:-N/A}"
echo "Node:         ${SLURM_NODELIST:-N/A}"
echo "Working Dir:  $(pwd)"
echo "Date:         $(date)"

task_id="${TASK_ID:-$SLURM_ARRAY_TASK_ID}"

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <param_path> <config_name> <train_script> <test_script> [--experiment_dir <dir>] [cli_args...]"
    exit 1
fi

param_path="$1"
config_name="$2"
train_script="$3"
test_script="$4"
FORWARD_ARGS="${*:5}"

echo ""
echo "Param Path:   $param_path"
echo "Config Name:  $config_name"
echo "Train Script: $train_script"
echo "Test Script:  $test_script"
echo "Forward Args: ${FORWARD_ARGS:-<none>}"

# ── Load sweep params for this task ──────────────────────────────────────────

echo ""
echo "--- Loading params for task $task_id ---"
sweep_params=$(
    python scripts/slurm/param_parser.py "$param_path" -i "$task_id"
)

if [[ -z "${sweep_params:-}" ]]; then
    echo "No params found for task $task_id - exiting"
    exit 1
fi
echo "Sweep Params: $sweep_params"

# ── Delegate to run_experiment.sh ────────────────────────────────────────────

echo ""
echo "--- Delegating to run_experiment.sh ---"
# shellcheck disable=SC2086
export TASK_ID="$task_id"
bash scripts/evaluation/run_experiment.sh \
    "$config_name" "$train_script" "$test_script" \
    $FORWARD_ARGS \
    $sweep_params
