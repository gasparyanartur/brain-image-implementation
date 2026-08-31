#!/bin/bash
# Run a full experiment pipeline: array(train+test) → aggregate.
#
# Each parameter combination is trained and tested in a single SLURM array task
# via run_sweep_instance.sh. When all tasks finish, an aggregate job collects results.
#
# Usage:
#   ./scripts/evaluation/run_experiment_sweep_slurm.sh <experiment_name> <param_path> <config_name> <train_script> <test_script> [cli_args...]
#
# Arguments:
#   experiment_name   Name for the experiment (used for the directory and SLURM job names).
#   param_path        Path to a param_parser JSON file defining the sweep.
#   config_name       Hydra config name passed to the training script.
#   train_script      Python training script (e.g. scripts/training/train_eeg.py).
#   test_script       Python evaluation script (e.g. scripts/evaluation/test_eeg.py).
#   cli_args          Extra Hydra overrides forwarded to the training script.
#
# Environment overrides (passed through to ssub):
#   SBATCH_GROUP, SBATCH_PARTITION, SBATCH_TIME, SBATCH_MEM, SBATCH_ACCOUNT, ...
#   TEST_HPARAMS      Space-separated list of dotted hparam keys to include in aggregate CSV.
#                     e.g. TEST_HPARAMS="model.lr model.eeg_encoder"
#   SBATCH_GROUP_AGGREGATE  SBATCH_GROUP override for the aggregate job (default: cpu).
#
# Examples:
#   ./scripts/evaluation/run_experiment_sweep_slurm.sh encoders scripts/params/encoders.json train_eeg \
#     scripts/training/train_eeg.py scripts/evaluation/test_eeg.py
#   TEST_HPARAMS="model.align_img_encoder model.eeg_encoder" \
#     ./scripts/evaluation/run_experiment_sweep_slurm.sh encoders scripts/params/encoders.json train_eeg \
#       scripts/training/train_eeg.py scripts/evaluation/test_eeg.py

set -euo pipefail

_root="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -f "$_root/.env" ]]; then
  set -a; source "$_root/.env"; set +a
fi

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <experiment_name> <param_path> <config_name> <train_script> <test_script> [cli_args...]"
    exit 1
fi

experiment_name="$1"
param_path="$2"
config_name="$3"
train_script="$4"
test_script="$5"
cli_args="${@:6}"

experiment_dir="experiments/${experiment_name}/$(date +%Y%m%d_%H%M%S)"

echo "=== Experiment Pipeline ==="
echo "  Name:         $experiment_name"
echo "  Dir:          $experiment_dir"
echo "  Params:       $param_path"
echo "  Config:       $config_name"
echo "  Train script: $train_script"
echo "  Test script:  $test_script"
echo "  Extra args:   ${cli_args:-<none>}"
echo ""

# ── Validate param file ───────────────────────────────────────────────────────

if [[ ! -f "$param_path" ]]; then
    echo "Error: param file not found: $param_path"
    exit 1
fi

param_count=$(python scripts/slurm/param_parser.py "$param_path" -s)
echo "Parameter combinations: $param_count"

if [[ "$param_count" -lt 1 ]]; then
    echo "Error: no parameter combinations found in $param_path"
    exit 1
fi

# ── Create experiment directory ───────────────────────────────────────────────

mkdir -p "$experiment_dir"
echo "Experiment directory: $experiment_dir"
echo ""

# ── Step 1: Submit SLURM array (train + test per task) ───────────────────────
# run_sweep_instance.sh is an orchestration script that calls run_singularity.sh
# internally, so we submit it with SSUB_NO_SINGULARITY=1 to avoid double-wrapping.

echo "--- Submitting sweep array ($param_count tasks: train + test each) ---"

sweep_instance_args=(
    "$param_path"
    "$config_name"
    "$train_script"
    "$test_script"
    "--experiment_dir" "$experiment_dir"
)
[[ -n "${cli_args:-}" ]] && sweep_instance_args+=($cli_args)

array_out=$(
    SBATCH_ARRAY="0-$((param_count - 1))" \
    SBATCH_OVERRIDE="--requeue --open-mode=append ${SBATCH_OVERRIDE:-}" \
    SSUB_NO_SINGULARITY=1 \
    ./scripts/slurm/ssub.sh "${experiment_name}" \
        bash scripts/evaluation/run_sweep_instance.sh \
            "${sweep_instance_args[@]}"
)
echo "$array_out"
array_job_id=$(echo "$array_out" | grep -oP '(?<=job )\d+')
echo "Array job ID: $array_job_id  ($param_count tasks)"
echo ""

# ── Step 2: Submit aggregate job (runs after all array tasks finish) ──────────

hparams_args=""
if [[ -n "${TEST_HPARAMS:-}" ]]; then
    hparams_args="--hparams ${TEST_HPARAMS}"
fi

echo "--- Submitting aggregate job (depends on array job $array_job_id) ---"
agg_out=$(
    SBATCH_OVERRIDE="--dependency=afterany:${array_job_id} ${SBATCH_OVERRIDE:-}" \
    SBATCH_GROUP="${SBATCH_GROUP_AGGREGATE:-cpu}" \
    ./scripts/slurm/ssub.sh "${experiment_name}" \
        python scripts/evaluation/aggregate_metrics.py \
            --experiment_dir "$experiment_dir" \
            --metrics_file_pattern '*test_metrics.csv' \
            $hparams_args
)
echo "$agg_out"
agg_job_id=$(echo "$agg_out" | grep -oP '(?<=job )\d+')
echo "Aggregate job ID: $agg_job_id"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────

echo "=== Pipeline submitted ==="
echo "  Array      ${array_job_id}  ($param_count tasks: train + test each)"
echo "  Aggregate  ${agg_job_id}   (depends on ${array_job_id})"
echo ""
echo "Results will be written to: ${experiment_dir}/aggregated_metrics.csv"
echo "Track first array task with: ./scripts/slurm/stalk.sh tail ${array_job_id}_0"
