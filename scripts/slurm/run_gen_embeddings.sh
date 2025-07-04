#!/bin/bash
#SBATCH --job-name=gen_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gen_embeddings/slurm/%j.out
#SBATCH --error=logs/gen_embeddings/slurm/%j.err
#SBATCH --account=berzelius-2025-35


# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"

image_path=${APPTAINER_IMAGE_PATH:-/home/x_artga/projdir/images/brain_2025_07_03.sif}

# Run the embedding generation script
apptainer exec --nv \
    --bind /proj:/proj \
    --bind /home:/home \
    --bind $PWD:/brain \
    --home /brain \
    $image_path \
    python /brain/scripts/gen_embeddings.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Embedding generation completed successfully"
else
    echo "Embedding generation failed with exit code $?"
    exit 1
fi

echo "Job completed at $(date)" 