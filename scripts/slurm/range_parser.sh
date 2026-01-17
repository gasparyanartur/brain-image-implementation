#!/usr/bin/env bash
# Author: Artur Gasparyan
# Date: 2026-01-17
# Licence: MIT

set -euo pipefail

for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
       echo "Generate a sequence of numbers from the given ranges."
       echo "Usage: $0 [start:end[::step]] ..."
       echo "Example: $0 1:3 5:10::2"
       echo "Output: 1 2 3 5 7 9"
       exit 0
    fi
done

for arg in "$@"; do
    step=1

    # Split on ::
    if [[ "$arg" == *"::"* ]]; then
        range="${arg%%::*}"
        step="${arg##*::}"
    else
        range="$arg"
    fi

    # Split start:end
    IFS=':' read -r start end <<< "$range"

    # Validate inputs
    if [[ -z "$start" || -z "$end" ]]; then
        echo "Invalid range: $arg" >&2
        exit 1
    fi

    if (( step == 0 )); then
        echo "Step cannot be zero: $arg" >&2
        exit 1
    fi

    # Forward or backward range
    if (( start <= end )); then
        for (( i=start; i<=end; i+=step )); do
            printf "%s " "$i"
        done
    else
        for (( i=start; i>=end; i-=step )); do
            printf "%s " "$i"
        done
    fi
done

echo
