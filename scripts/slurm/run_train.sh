#!/usr/bin/env bash
# Thin wrapper: submit a training job via ssub.sh
# Usage: run_train.sh <job_name> <train_script> [args...]
train_script="${2:?train_script required}"
if [[ ! -f "$train_script" ]]; then
  echo "Error: train_script does not exist: $train_script" >&2
  exit 1
fi
exec "$(dirname "$0")/ssub.sh" "${1:?job_name required}" --wrap python "$train_script" "${@:3}"