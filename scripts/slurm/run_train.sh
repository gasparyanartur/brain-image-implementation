#!/bin/bash
#SBATCH --job-name=train_comm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/train_comm/%j.out
#SBATCH --account=Berzelius-2025-278

train_script="$1"
echo "Train scripts: $train_script"
if [ -z "$train_script" ]; then
    echo "No train script provided"
    exit 1
fi

if [ ! -f "$train_script" ]; then
    echo "Train script $train_script does not exist"
    exit 1
fi

CLI_ARGS="${@:2}"
echo "CLI_ARGS: $CLI_ARGS"

./scripts/container/run_singularity.sh \
    python $train_script $CLI_ARGS 

if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    exit 1
fi

echo "Job completed at $(date)" 