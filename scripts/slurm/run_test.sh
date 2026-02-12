#!/bin/bash
#SBATCH --job-name=train_eeg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/test_eeg/%j.out
#SBATCH --account=Berzelius-2025-278

test_script=$1
echo "Test script: $test_script"
echo "Image path: $image_path"

# Other args
CLI_ARGS="${@:2}"
echo "CLI_ARGS: $CLI_ARGS"

./scripts/container/run_singularity.sh \
    python $test_script $CLI_ARGS 

if [ $? -eq 0 ]; then
    echo "Testing completed successfully"
else
    echo "Testing failed with exit code $?"
    exit 1
fi

echo "Job completed at $(date)" 