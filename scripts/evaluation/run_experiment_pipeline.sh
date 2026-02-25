#!/bin/bash
# Run a full experiment pipeline: train → test → aggregate.
#
# Usage:
#   ./scripts/evaluation/run_experiment_pipeline.sh <experiment_name> <param_path> <config_name> [cli_args...]
#
# Arguments:
#   experiment_name   Name for the experiment (used for the directory and SLURM job names).
#   param_path        Path to a param_parser JSON file defining the sweep.
#   config_name       Hydra config name passed to train_eeg.py (--config-name=<config_name>).
#   cli_args          Extra Hydra overrides forwarded to the training script.
#
# Environment overrides (passed through to ssub):
#   SBATCH_GROUP, SBATCH_PARTITION, SBATCH_TIME, SBATCH_MEM, SBATCH_ACCOUNT, ...
#   TEST_HPARAMS      Space-separated list of dotted hparam keys to include in aggregate CSV.
#                     e.g. TEST_HPARAMS="model.lr model.eeg_encoder"
#
# Examples:
#   ./scripts/evaluation/run_experiment_pipeline.sh encoders scripts/slurm/params/encoders.json train_eeg
#   TEST_HPARAMS="model.align_img_encoder model.eeg_encoder" \
#     ./scripts/evaluation/run_experiment_pipeline.sh encoders scripts/slurm/params/encoders.json train_eeg

set -euo pipefail

_root="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -f "$_root/.env" ]]; then
  set -a; source "$_root/.env"; set +a
fi

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <experiment_name> <param_path> <config_name> [cli_args...]"
    exit 1
fi

experiment_name="$1"
param_path="$2"
config_name="$3"
cli_args="${@:4}"

experiment_dir="experiments/${experiment_name}"

echo "=== Experiment Pipeline ==="
echo "  Name:        $experiment_name"
echo "  Dir:         $experiment_dir"
echo "  Params:      $param_path"
echo "  Config:      $config_name"
echo "  Extra args:  ${cli_args:-<none>}"
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

# ── Step 1: Submit one training job per parameter combination ─────────────────
# Params are baked in at submission time — no wrapper script needed at runtime.
# --requeue:          requeue automatically on node failure / preemption.
# --open-mode=append: requeued jobs append to existing log rather than truncating.

echo "--- Submitting $param_count training jobs ---"
train_job_ids=()

for (( i=0; i<param_count; i++ )); do
    sweep_params=$(python scripts/slurm/param_parser.py "$param_path" -i "$i")

    job_out=$(
        SBATCH_OVERRIDE="--requeue --open-mode=append ${SBATCH_OVERRIDE:-}" \
        ./scripts/slurm/ssub.sh "${experiment_name}" \
            python scripts/training/train_eeg.py \
                --config-name="$config_name" \
                trainer.log_dir="$experiment_dir" \
                $sweep_params \
                $cli_args
    )
    echo "$job_out"
    job_id=$(echo "$job_out" | grep -oP '(?<=job )\d+')
    train_job_ids+=("$job_id")
done

# Build colon-separated dependency string: afterany:id1:id2:...
# afterany: test runs even if some training jobs failed, so partial results are tested.
train_dep=$(IFS=:; echo "afterany:${train_job_ids[*]}")
echo ""
echo "Train job IDs: ${train_job_ids[*]}"

# ── Step 2: Submit test job (runs after all training jobs finish) ─────────────

echo ""
echo "--- Submitting test job (depends on all training jobs) ---"
test_out=$(
    SBATCH_OVERRIDE="--dependency=${train_dep} ${SBATCH_OVERRIDE:-}" \
    ./scripts/slurm/ssub.sh "${experiment_name}" \
        bash scripts/evaluation/test_all_experiments.sh \
            scripts/evaluation/test_eeg.py \
            "$experiment_dir"
)
echo "$test_out"
test_job_id=$(echo "$test_out" | grep -oP '(?<=job )\d+')
echo "Test job ID: $test_job_id"
echo ""

# ── Step 3: Submit aggregate job (runs after test job finishes) ───────────────

hparams_args=""
if [[ -n "${TEST_HPARAMS:-}" ]]; then
    hparams_args="--hparams ${TEST_HPARAMS}"
fi

echo "--- Submitting aggregate job (depends on test job) ---"
agg_out=$(
    SBATCH_OVERRIDE="--dependency=afterany:${test_job_id} ${SBATCH_OVERRIDE:-}" \
    SBATCH_GROUP="${SBATCH_GROUP_AGGREGATE:-cpu}" \
    ./scripts/slurm/ssub.sh "${experiment_name}" \
        python scripts/evaluation/aggregate_metrics.py \
            --experiment_dir "$experiment_dir" \
            $hparams_args
)
echo "$agg_out"
agg_job_id=$(echo "$agg_out" | grep -oP '(?<=job )\d+')
echo "Aggregate job ID: $agg_job_id"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────

echo "=== Pipeline submitted ==="
echo "  Train      [${train_job_ids[*]}]  ($param_count jobs)"
echo "  Test       ${test_job_id}         (depends on all train jobs)"
echo "  Aggregate  ${agg_job_id}          (depends on ${test_job_id})"
echo ""
echo "Results will be written to: ${experiment_dir}/aggregated_metrics.csv"
echo "Track first train job with: ./scripts/slurm/stalk.sh tail ${train_job_ids[0]}"
