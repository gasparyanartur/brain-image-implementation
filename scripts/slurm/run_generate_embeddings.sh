#!/bin/bash
#SBATCH --job-name=generate_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/generate_embeddings/%j.out
#SBATCH --account=berzelius-2025-35


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

# Get CLI arguments (all arguments passed to this script)
CLI_ARGS="$@"

echo "CLI_ARGS: $CLI_ARGS"

# Run the embedding generation script
./scripts/container/run_singularity.sh \
    python /workspace/scripts/generate_embeddings.py $CLI_ARGS

# Check exit status
if [ $? -eq 0 ]; then
    echo "Embedding generation completed successfully"
else
    echo "Embedding generation failed with exit code $?"
    exit 1
fi

echo "Job completed at $(date)" 