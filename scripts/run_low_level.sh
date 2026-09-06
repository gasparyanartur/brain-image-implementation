#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:-experiments/low_level/$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi
config_name="${LOW_LEVEL_CONFIG:-train_low_level}"

python scripts/training/train_low_level.py \
  "--config-name=${config_name}" \
  "trainer.log_dir=${run_dir}" \
  "$@"

python scripts/evaluation/test_low_level.py "${run_dir}" \
  "--config_name=${config_name}"
