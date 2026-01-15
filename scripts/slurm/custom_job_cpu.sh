#!/bin/bash
#SBATCH --job-name=custom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/custom_cpu/%A_%a.out
#SBATCH --account=Berzelius-2025-278

CLI_ARGS="$@"

echo "CLI_ARGS: $CLI_ARGS"
./scripts/container/run_singularity.sh $CLI_ARGS