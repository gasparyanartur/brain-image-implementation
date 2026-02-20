#!/usr/bin/env bash
# Thin wrapper: submit a CPU wrap job via ssub.sh
# Usage: custom_job_cpu.sh <job_name> <command...>
SBATCH_GROUP=cpu exec "$(dirname "$0")/ssub.sh" "${1:?job_name required}" --wrap "${@:2}"
./scripts/container/run_singularity.sh $CLI_ARGS