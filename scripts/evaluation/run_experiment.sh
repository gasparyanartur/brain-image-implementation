#!/bin/bash
# Train one configuration then immediately evaluate it.
# Can be run standalone or called by run_sweep_instance.sh.
#
# Usage:
#   ./scripts/evaluation/run_experiment.sh \
#     <config_name> <train_script> <test_script> [--experiment_dir <dir>] [cli_args...]
#
# Args:
#   config_name     Hydra config name passed to the train script.
#   train_script    Python training script (e.g. scripts/training/train_eeg.py).
#   test_script     Python evaluation script (e.g. scripts/evaluation/test_eeg.py).
#
# Options:
#   --experiment_dir <dir>   Directory for training logs/checkpoints (default: logs/experiments).
#
# cli_args: any remaining arguments are forwarded verbatim as Hydra overrides to the training script.
#
# Env overrides:
#   TASK_ID   When set, used to locate the run directory after training (set by run_sweep_instance.sh).

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <config_name> <train_script> <test_script> [--experiment_dir <dir>] [cli_args...]"
    exit 1
fi

config_name="$1"
train_script="$2"
test_script="$3"
shift 3

experiment_dir="logs/experiments"
if [[ "${1:-}" == "--experiment_dir" ]]; then
    experiment_dir="$2"
    shift 2
fi

CLI_ARGS="${*}"

echo "Config Name:     $config_name"
echo "Train Script:    $train_script"
echo "Test Script:     $test_script"
echo "Experiment Dir:  $experiment_dir"
echo "CLI Args:        ${CLI_ARGS:-<none>}"
echo ""

mkdir -p "$experiment_dir"

# ── Train ─────────────────────────────────────────────────────────────────────

echo "--- Training ($train_script) ---"
# shellcheck disable=SC2086
./scripts/container/run_singularity.sh \
    python "$train_script" \
        --config-name="$config_name" \
        trainer.log_dir="$experiment_dir" \
        $CLI_ARGS

# ── Find run directory ────────────────────────────────────────────────────────
#
# The trainer encodes SLURM_ARRAY_JOB_ID and SLURM_ARRAY_TASK_ID into the run
# directory name: {run_name}-{array_job_id}_{task_id}-{timestamp}.
# We match on that pattern; fall back to the newest directory otherwise.

echo ""
echo "--- Locating run directory ---"

task_id="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "$task_id" ]]; then
    run_dir=$(
        ls -td "${experiment_dir}/"*"-${SLURM_ARRAY_JOB_ID}_${task_id}-"* 2>/dev/null | head -1
    )
else
    run_dir=$(
        ls -td "${experiment_dir}/"*/ 2>/dev/null | head -1
    )
    run_dir="${run_dir%/}"
fi

if [[ -z "${run_dir:-}" ]]; then
    echo "Error: could not find run directory in $experiment_dir — skipping test"
    exit 1
fi
echo "Run directory: $run_dir"

# ── Test ──────────────────────────────────────────────────────────────────────

echo ""
echo "--- Testing ($test_script) ---"
./scripts/container/run_singularity.sh python "$test_script" "$run_dir"

echo ""
echo "Done."
