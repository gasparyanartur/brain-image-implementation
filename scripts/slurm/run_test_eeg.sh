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