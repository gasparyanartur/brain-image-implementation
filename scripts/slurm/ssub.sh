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

Resource defaults are loaded from scripts/slurm/ssub_groups.conf (edit that file
  to add or modify groups). Set SBATCH_GROUP to select a group (default: gpu).

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

# Load resource defaults from ssub_groups.conf
_groups_conf="$_root/scripts/slurm/ssub_groups.conf"
read -r _cpus_def _mem_def _time_def _gpus_def _qos_def < <(
  awk -v grp="$group" '!/^[[:space:]]*#/ && $1 == grp { print $2, $3, $4, $5, $6; exit }' "$_groups_conf"
)
if [[ -z "${_cpus_def:-}" ]]; then
  valid_groups=$(awk '!/^[[:space:]]*#/ && NF { print $1 }' "$_groups_conf" | paste -sd ', ')
  echo "Error: unknown SBATCH_GROUP '$group'. Valid groups: $valid_groups" >&2
  exit 2
fi

if [[ "$_gpus_def" == "-" ]]; then
  partition="${SBATCH_PARTITION:-${SBATCH_CPU_PARTITION:-}}"
  gpus="${SBATCH_GPUS:-}"
else
  partition="${SBATCH_PARTITION:-${SBATCH_GPU_PARTITION:-}}"
  gpus="${SBATCH_GPUS:-$_gpus_def}"
fi
cpus="${SBATCH_CPUS_PER_TASK:-$_cpus_def}"
mem="${SBATCH_MEM:-$_mem_def}"
time="${SBATCH_TIME:-$_time_def}"
qos=""
[[ "$_qos_def" != "-" ]] && qos="$_qos_def"

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
