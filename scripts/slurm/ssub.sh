#!/usr/bin/env bash
set -euo pipefail

# Auto-load .env from the repo root if present.
_root="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -f "$_root/.env" ]]; then
  set -a; source "$_root/.env"; set +a
fi

usage() {
  cat <<'EOF'
Usage: ssub <job_name> [-t] [--dry-run] <command...>

  job_name    Slurm job name (also used for log directory under logs/slurm/)
  -t          Tail the job log after submission (via stalk.sh tail <job_id>)
  command...  Any command to run inside the Singularity container

Resource defaults (SBATCH_GROUP=gpu [default] | gpu-light | cpu | cpu-light):
  gpu:       --cpus-per-task=32 --mem=128G --time=1-00:00:00 --gpus=1
  gpu-light: --cpus-per-task=8  --mem=32G  --time=01:00:00   --gpus=1  --qos=devel
  cpu:       --cpus-per-task=32 --mem=128G --time=1-00:00:00
  cpu-light: --cpus-per-task=8  --mem=32G  --time=01:00:00   --qos=devel

Environment overrides:
  SBATCH_GROUP, SBATCH_GPU_PARTITION, SBATCH_CPU_PARTITION, SBATCH_PARTITION
  SBATCH_NODES, SBATCH_NTASKS, SBATCH_CPUS_PER_TASK, SBATCH_MEM, SBATCH_TIME
  SBATCH_GPUS, SBATCH_ARRAY, SBATCH_ACCOUNT
  SBATCH_OVERRIDE   extra sbatch flags, e.g. "--qos=debug --constraint=foo"
  SLURM_LOG_DIR     log root (default: logs/slurm)
  SSUB_DRY_RUN=1    print sbatch command without submitting
  SSUB_NO_SINGULARITY=1  skip container wrapping
EOF
}

print_command_quoted() {
  local out="" part
  for part in "$@"; do out+="$(printf '%q ' "$part")"; done
  printf '%s\n' "${out% }"
}

override_has_array() {
  local s="${1:-}"
  [[ -z "$s" ]] && return 1
  [[ "$s" =~ (^|[[:space:]])(--array(=|[[:space:]])|-a([[:space:]]|$)) ]]
}

# ── Args ──────────────────────────────────────────────────────────────────────

if [[ $# -lt 2 ]]; then usage; exit 2; fi
[[ "${1:-}" == "--help" || "${1:-}" == "-h" ]] && { usage; exit 0; }

job_name="$1"; shift
dry_run="${SSUB_DRY_RUN:-0}"
tail_job=0
[[ "${1:-}" == "--dry-run" ]] && { dry_run=1; shift; }
[[ "${1:-}" == "-t"        ]] && { tail_job=1; shift; }
[[ $# -lt 1 ]] && { echo "Error: a command is required" >&2; usage; exit 2; }
wrap_cmd=("$@")

# ── Group & resource defaults ─────────────────────────────────────────────────

group="${SBATCH_GROUP:-gpu}"
[[ "$group" =~ ^(gpu|gpu-light|cpu|cpu-light)$ ]] || { echo "Error: SBATCH_GROUP must be gpu, gpu-light, cpu, or cpu-light" >&2; exit 2; }

case "$group" in
  gpu)
    partition="${SBATCH_PARTITION:-${SBATCH_GPU_PARTITION:-}}"
    gpus="${SBATCH_GPUS:-1}"
    cpus="${SBATCH_CPUS_PER_TASK:-32}"
    mem="${SBATCH_MEM:-128G}"
    time="${SBATCH_TIME:-1-00:00:00}"
    qos=""
    ;;
  gpu-light)
    partition="${SBATCH_PARTITION:-${SBATCH_GPU_PARTITION:-}}"
    gpus="${SBATCH_GPUS:-1}"
    cpus="${SBATCH_CPUS_PER_TASK:-8}"
    mem="${SBATCH_MEM:-32G}"
    time="${SBATCH_TIME:-01:00:00}"
    qos="devel"
    ;;
  cpu)
    partition="${SBATCH_PARTITION:-${SBATCH_CPU_PARTITION:-}}"
    gpus=""
    cpus="${SBATCH_CPUS_PER_TASK:-32}"
    mem="${SBATCH_MEM:-128G}"
    time="${SBATCH_TIME:-1-00:00:00}"
    qos=""
    ;;
  cpu-light)
    partition="${SBATCH_PARTITION:-${SBATCH_CPU_PARTITION:-}}"
    gpus=""
    cpus="${SBATCH_CPUS_PER_TASK:-8}"
    mem="${SBATCH_MEM:-32G}"
    time="${SBATCH_TIME:-01:00:00}"
    qos="devel"
    ;;
esac
[[ -n "${SBATCH_GPUS:-}" ]] && gpus="$SBATCH_GPUS"

nodes="${SBATCH_NODES:-1}"
ntasks="${SBATCH_NTASKS:-1}"

[[ -z "$partition" ]] && { echo "Error: no partition. Set SBATCH_GPU_PARTITION / SBATCH_CPU_PARTITION / SBATCH_PARTITION." >&2; exit 2; }

# ── Logs ──────────────────────────────────────────────────────────────────────

log_dir="${SLURM_LOG_DIR:-logs/slurm}/${job_name}"
mkdir -p "$log_dir"

if [[ -n "${SBATCH_ARRAY:-}" ]] || override_has_array "${SBATCH_OVERRIDE:-}"; then
  is_array=1
  out_path="${SBATCH_OUTPUT:-${log_dir}/%A_%a.out}"
  err_path="${SBATCH_ERROR:-${log_dir}/%A_%a.err}"
else
  is_array=0
  out_path="${SBATCH_OUTPUT:-${log_dir}/%j.out}"
  err_path="${SBATCH_ERROR:-${log_dir}/%j.err}"
fi

# ── Build sbatch args ─────────────────────────────────────────────────────────

sbatch_args=(
  --job-name="$job_name"
  --nodes="$nodes" --ntasks="$ntasks" --cpus-per-task="$cpus"
  --mem="$mem" --time="$time" --partition="$partition"
  --output="$out_path" --error="$err_path"
)
[[ -n "${SBATCH_ACCOUNT:-}" ]] && sbatch_args+=(--account="$SBATCH_ACCOUNT")
[[ -n "${SBATCH_ARRAY:-}"   ]] && sbatch_args+=(--array="$SBATCH_ARRAY")
[[ -n "$gpus"               ]] && sbatch_args+=(--gpus="$gpus")
[[ -n "${qos:-}"            ]] && sbatch_args+=(--qos="$qos")
if [[ -n "${SBATCH_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  sbatch_args+=( $SBATCH_OVERRIDE )
fi

# ── Wrap & submit ─────────────────────────────────────────────────────────────

[[ "${SSUB_NO_SINGULARITY:-0}" != "1" ]] && wrap_cmd=(./scripts/container/run_singularity.sh "${wrap_cmd[@]}")
wrap_str="${wrap_cmd[*]}"
final_args=("${sbatch_args[@]}" --wrap "$wrap_str")

echo "[ssub] group=$group array=$is_array | logs: $log_dir"
echo "[ssub] cmd: $wrap_str"

if [[ "$dry_run" == "1" ]]; then
  echo "[ssub] dry-run: $(print_command_quoted sbatch "${final_args[@]}")"
  exit 0
fi

submit_out=$(sbatch "${final_args[@]}")
echo "[ssub] $submit_out"

if [[ "$tail_job" == "1" ]]; then
  job_id=$(grep -oP '(?<=job )\d+' <<< "$submit_out")
  echo "[ssub] tailing job $job_id"
  exec ./scripts/slurm/stalk.sh tail "$job_id"
fi
