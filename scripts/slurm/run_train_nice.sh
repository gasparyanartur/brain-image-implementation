#!/bin/bash
#SBATCH --job-name=train_nice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train_nice/slurm/%j.out
#SBATCH --account=berzelius-2025-35


# Print job information
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

# Run the training script
./scripts/container/run_singularity.sh \
    python /workspace/scripts/train_nice.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "NICE training completed successfully"
else
    echo "NICE training failed with exit code $?"
    exit 1
fi

echo "Job completed at $(date)" 