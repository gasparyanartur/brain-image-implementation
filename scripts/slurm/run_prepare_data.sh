#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=berzelius-cpu
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/prepare_data/%j.out
#SBATCH --account=Berzelius-2025-278


echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"

image_path=${APPTAINER_IMAGE_PATH}
# if image_path is not set, use the latest image path
if [ -z "$image_path" ]; then
    image_path=$(ls -t images/brain_*.sif | head -n 1)
fi

dataset=$1
echo "Dataset: $dataset"

CLI_ARGS="${@:2}"
echo "CLI_ARGS: $CLI_ARGS"

cmd="/workspace/scripts/data/${dataset}/prepare.sh $CLI_ARGS"
echo "Preparing data with command: $cmd"

./scripts/container/run_singularity.sh $cmd

if [ $? -eq 0 ]; then
    echo "Prepared data successfully"
else
    echo "Data preparation failed with exit code $?"
    exit 1
fi

echo "Job completed at $(date)" 