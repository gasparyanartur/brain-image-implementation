#!/usr/bin/env bash
# Thin wrapper: submit a GPU wrap job via ssub.sh
# Usage: custom_job.sh <job_name> <command...>
exec "$(dirname "$0")/ssub.sh" "${1:?job_name required}" --wrap "${@:2}"