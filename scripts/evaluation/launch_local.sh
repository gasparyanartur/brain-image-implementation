#!/bin/bash
# Run a single parameter configuration locally (no SLURM, no Singularity),
# or aggregate results from a completed experiment.
#
# Usage:
#   ./scripts/evaluation/launch_local.sh <experiment_name> <task_id|AGG> <param_path> <config_name> <train_script> <test_script> [cli_args...]
#
# Arguments:
#   experiment_name Name for the experiment (results go to experiments/<name>/<timestamp>_task<id>).
#   task_id         0-based index into the parameter combinations defined in param_path.
#                   Pass AGG to skip training and run aggregation on the experiment dir instead.
#   param_path      Path to a param_parser JSON file defining the sweep.
#   config_name     Hydra config name passed to the training script.
#   train_script    Python training script (e.g. scripts/training/train_eeg.py).
#   test_script     Python evaluation script (e.g. scripts/evaluation/test_eeg.py).
#   cli_args        Extra Hydra overrides forwarded verbatim to the training script.
#
# Examples:
#   # Run the 3rd (0-based) encoder configuration from the text_vs_img sweep:
#   ./scripts/evaluation/launch_local.sh text_vs_img 2 scripts/params/text_vs_img_encoders.json \
#       train_eeg_align_text scripts/training/train_eeg.py scripts/evaluation/test_eeg.py
#
#   # Aggregate all completed runs in an experiment:
#   ./scripts/evaluation/launch_local.sh text_vs_img AGG scripts/params/text_vs_img_encoders.json \
#       train_eeg_align_text scripts/training/train_eeg.py scripts/evaluation/test_eeg.py
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

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <experiment_name> <task_id|AGG> <param_path> [<config_name> <train_script> <test_script>] [cli_args...]"
    echo ""
    echo "  experiment_name  Name for the experiment (results go to experiments/<name>/...)"
    echo "  task_id|AGG      0-based index of the parameter combination, or AGG to aggregate"
    echo "  param_path       Path to the param_parser JSON file"
    echo "  config_name      Hydra config name (not needed for AGG)"
    echo "  train_script     Python training script (not needed for AGG)"
    echo "  test_script      Python evaluation script (not needed for AGG)"
    exit 1
fi

experiment_name="$1"
task_id="$2"
param_path="$3"
config_name="${4:-}"
train_script="${5:-}"
test_script="${6:-}"
[[ $# -ge 6 ]] && shift 6 || shift $#
extra_cli_args="${*:-}"

# ── AGG mode ──────────────────────────────────────────────────────────────────

if [[ "$task_id" == "AGG" ]]; then
    agg_dir="experiments/${experiment_name}"
    if [[ ! -d "$agg_dir" ]]; then
        echo "Error: experiment directory not found: $agg_dir"
        exit 1
    fi
    echo "=== Aggregating results in: $agg_dir ==="
    hparams_args=""
    if [[ -n "${TEST_HPARAMS:-}" ]]; then
        hparams_args="--hparams ${TEST_HPARAMS}"
    fi
    # shellcheck disable=SC2086
    python scripts/evaluation/aggregate_metrics.py \
        --experiment_dir "$agg_dir" \
        $hparams_args
    echo "Done. Results in: ${agg_dir}/aggregated_metrics.csv"
    exit 0
fi

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
