#!/usr/bin/env bash
set -euo pipefail

# Auto-load .env from the repo root if present.
_root="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -f "$_root/.env" ]]; then
  set -a; source "$_root/.env"; set +a
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/slurm/ssub.sh <job_name> <script_path> [--dry-run] [script args...]
  scripts/slurm/ssub.sh <job_name> --wrap <command...>

Modes:
  script mode   Submit an existing Slurm job script via sbatch.
  wrap mode     Submit an arbitrary command via sbatch --wrap (no job script needed).

Required args:
  job_name            Slurm job name (also used for log directory)
  script_path         Job script to submit (passed to sbatch), or --wrap

Defaults (by group):
  SBATCH_GROUP=gpu (default) or cpu

  cpu:
    --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=1-00:00:00
    --partition=$SBATCH_CPU_PARTITION
  gpu:
    --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --time=1-00:00:00 --gpus=1
    --partition=$SBATCH_GPU_PARTITION

Environment overrides:
  SLURM_LOG_DIR          (default: logs/slurm) -> creates $SLURM_LOG_DIR/<job_name>
  SBATCH_ACCOUNT         -> passes --account=...
  SBATCH_NODES           -> --nodes=...
  SBATCH_NTASKS          -> --ntasks=...
  SBATCH_CPUS_PER_TASK   -> --cpus-per-task=...
  SBATCH_MEM             -> --mem=...
  SBATCH_TIME            -> --time=...
  SBATCH_PARTITION       -> --partition=... (overrides group partition)
  SBATCH_GPUS            -> --gpus=...
  SBATCH_ARRAY           -> --array=... (also affects log pattern)

Free-form overrides:
  SBATCH_OVERRIDE         Extra sbatch flags (split on spaces), appended last.
                          Example: SBATCH_OVERRIDE="--qos=debug --constraint=foo"

Local testing:
  Pass --dry-run as 3rd arg, or set SSUB_DRY_RUN=1.

Singularity:
  Wrap mode prepends ./scripts/container/run_singularity.sh automatically.
  Set SSUB_NO_SINGULARITY=1 to disable.

Notes:
  - Array detection is automatic: we inspect SBATCH_ARRAY, SBATCH_OVERRIDE, AND any
    '#SBATCH --array ...' directive in the job script.
  - In script mode, ssub respects '#SBATCH ...' directives by default: it only
    passes a flag if you explicitly override it via SBATCH_* env vars.
  - In wrap mode, ssub supplies group defaults (cpu/gpu) and then applies env
    overrides.

Logging:
  Non-array: $SLURM_LOG_DIR/<job_name>/%j.out and %j.err
  Array:     $SLURM_LOG_DIR/<job_name>/%A_%a.out and %A_%a.err
EOF
}

print_command_quoted() {
  # Print a command with shell-escaped args for copy/paste.
  local out=""
  local part
  for part in "$@"; do
    out+="$(printf '%q ' "$part")"
  done
  printf '%s\n' "${out% }"
}

script_has_sbatch_directive() {
  local script="$1"
  local flag="$2" # e.g. --array, --output
  # Match '#SBATCH   --flag' or '#SBATCH --flag=...'
  grep -Eq "^[[:space:]]*#SBATCH[[:space:]]+${flag}([[:space:]]|=|$)" "$script"
}

detect_array_from_script() {
  local script="$1"
  if script_has_sbatch_directive "$script" "--array"; then
    return 0
  fi
  # Also support short form -a
  if grep -Eq "^[[:space:]]*#SBATCH[[:space:]]+-a([[:space:]]|$)" "$script"; then
    return 0
  fi
  return 1
}

override_has_array() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    return 1
  fi
  if [[ "$s" =~ (^|[[:space:]])(--array(=|[[:space:]])|-a([[:space:]]|$)) ]]; then
    return 0
  fi
  return 1
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

job_name="${1:-}"
script_or_wrap="${2:-}"
shift 2

if [[ -z "$job_name" ]]; then
  echo "Error: job_name is required" >&2
  exit 2
fi

mode="script"
script_path=""
wrap_cmd=()
dry_run="${SSUB_DRY_RUN:-0}"

if [[ "$script_or_wrap" == "--wrap" ]]; then
  mode="wrap"
  if [[ $# -lt 1 ]]; then
    echo "Error: --wrap requires a command" >&2
    usage
    exit 2
  fi
  wrap_cmd=("$@")
else
  script_path="$script_or_wrap"
  if [[ ! -f "$script_path" ]]; then
    echo "Error: script_path does not exist: $script_path" >&2
    exit 2
  fi
  if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift 1
  fi
fi

group="${SBATCH_GROUP:-gpu}"
if [[ "$group" != "cpu" && "$group" != "gpu" ]]; then
  echo "Error: SBATCH_GROUP must be 'cpu' or 'gpu' (got: $group)" >&2
  exit 2
fi

log_root="${SLURM_LOG_DIR:-logs/slurm}"
log_dir="${log_root%/}/${job_name}"
mkdir -p "$log_dir"

nodes_default="1"
ntasks_default="1"
cpus_per_task_default="32"
time_limit_default="1-00:00:00"

mem_default=""
partition_default=""
gpus_default=""

if [[ "$group" == "cpu" ]]; then
  mem_default="128G"
  partition_default="${SBATCH_CPU_PARTITION:-}"
  gpus_default=""
else
  mem_default="128G"
  partition_default="${SBATCH_GPU_PARTITION:-}"
  gpus_default="1"
fi

nodes="${SBATCH_NODES:-$nodes_default}"
ntasks="${SBATCH_NTASKS:-$ntasks_default}"
cpus_per_task="${SBATCH_CPUS_PER_TASK:-$cpus_per_task_default}"
time_limit="${SBATCH_TIME:-$time_limit_default}"

mem="${SBATCH_MEM:-$mem_default}"

# Script mode should not override a script's partition unless explicitly asked.
partition=""
if [[ "$mode" == "wrap" ]]; then
  partition="${SBATCH_PARTITION:-$partition_default}"
else
  partition="${SBATCH_PARTITION:-}"
fi

# Important: only supply a default GPU request in wrap mode.
gpus=""
if [[ -n "${SBATCH_GPUS:-}" ]]; then
  gpus="${SBATCH_GPUS}"
elif [[ "$mode" == "wrap" && "$group" == "gpu" ]]; then
  gpus="$gpus_default"
fi

if [[ "$mode" == "wrap" && -z "$partition" ]]; then
  if [[ "$group" == "cpu" ]]; then
    echo "Error: missing partition. Set SBATCH_CPU_PARTITION or SBATCH_PARTITION." >&2
  else
    echo "Error: missing partition. Set SBATCH_GPU_PARTITION or SBATCH_PARTITION." >&2
  fi
  exit 2
fi
if [[ "$mode" == "script" && -n "${SBATCH_PARTITION:-}" && -z "$partition" ]]; then
  echo "Error: SBATCH_PARTITION is set but empty" >&2
  exit 2
fi

is_array=0
array_reason=""
if [[ -n "${SBATCH_ARRAY:-}" ]]; then
  is_array=1
  array_reason="SBATCH_ARRAY is set"
elif override_has_array "${SBATCH_OVERRIDE:-}"; then
  is_array=1
  array_reason="SBATCH_OVERRIDE contains --array/-a"
elif [[ "$mode" == "script" ]] && detect_array_from_script "$script_path"; then
  is_array=1
  array_reason="job script has #SBATCH --array/-a"
else
  is_array=0
  array_reason="no array directive detected"
fi

out_path=""
err_path=""

if [[ $is_array -eq 1 ]]; then
  out_path="${SBATCH_OUTPUT:-${log_dir}/%A_%a.out}"
  err_path="${SBATCH_ERROR:-${log_dir}/%A_%a.err}"
else
  out_path="${SBATCH_OUTPUT:-${log_dir}/%j.out}"
  err_path="${SBATCH_ERROR:-${log_dir}/%j.err}"
fi

declare -a sbatch_args

sbatch_args+=(--job-name="$job_name")

if [[ "$mode" == "wrap" ]]; then
  # Wrap mode: no job script, so supply all resource flags.
  sbatch_args+=(--nodes="$nodes" --ntasks="$ntasks" --cpus-per-task="$cpus_per_task")
  sbatch_args+=(--mem="$mem" --time="$time_limit" --partition="$partition")
else
  # Script mode: only override what's explicitly set; let the script's #SBATCH directives handle the rest.
  [[ -n "${SBATCH_NODES:-}" ]]         && sbatch_args+=(--nodes="$nodes")
  [[ -n "${SBATCH_NTASKS:-}" ]]        && sbatch_args+=(--ntasks="$ntasks")
  [[ -n "${SBATCH_CPUS_PER_TASK:-}" ]] && sbatch_args+=(--cpus-per-task="$cpus_per_task")
  [[ -n "${SBATCH_MEM:-}" ]]           && sbatch_args+=(--mem="$mem")
  [[ -n "${SBATCH_TIME:-}" ]]          && sbatch_args+=(--time="$time_limit")
  [[ -n "${SBATCH_PARTITION:-}" ]]     && sbatch_args+=(--partition="$partition")
fi

sbatch_args+=(--output="$out_path" --error="$err_path")

if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  sbatch_args+=(--account="$SBATCH_ACCOUNT")
fi

if [[ -n "${SBATCH_ARRAY:-}" ]]; then
  sbatch_args+=(--array="$SBATCH_ARRAY")
fi

if [[ -n "$gpus" ]]; then
  sbatch_args+=(--gpus="$gpus")
fi

if [[ -n "${SBATCH_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  override_args=( $SBATCH_OVERRIDE )
  sbatch_args+=("${override_args[@]}")
fi

echo "[ssub] mode=$mode group=$group array=$is_array ($array_reason)"
echo "[ssub] logs: out=$out_path err=$err_path"

submit_out=""
if [[ "$mode" == "wrap" ]]; then
  if [[ "${SSUB_NO_SINGULARITY:-0}" != "1" ]]; then
    wrap_cmd=(./scripts/container/run_singularity.sh "${wrap_cmd[@]}")
  fi
  # Turn the wrap_cmd array into a single string for sbatch --wrap.
  wrap_str="${wrap_cmd[*]}"
  final_args=("${sbatch_args[@]}" --wrap "$wrap_str")
else
  final_args=("${sbatch_args[@]}" "$script_path" "$@")
fi

if [[ "$dry_run" == "1" ]]; then
  echo "[ssub] dry-run cmd: $(print_command_quoted sbatch "${final_args[@]}")" 
  exit 0
fi

echo "[ssub] submitting $mode=$( [[ "$mode" == "wrap" ]] && echo "$wrap_str" || echo "$script_path" )"
submit_out=$(sbatch "${final_args[@]}")
echo "[ssub] $submit_out"
