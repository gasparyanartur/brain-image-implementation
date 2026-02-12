#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=berzelius-cpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/prepare_data/%j.out
#SBATCH --account=Berzelius-2025-278

script_dir=$(dirname "$0")
dataset=$1
shift
"$script_dir/run_prepare_data.sh" "$dataset" --modality eeg "$@"
