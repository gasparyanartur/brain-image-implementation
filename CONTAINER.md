# Container Usage

This project provides both Docker and Singularity/Apptainer container support for reproducible environments.

## Docker

### Building the Docker Image

```bash
# Using the build script
./build_docker.sh

# Or manually
docker build -t brain-image-implementation:latest .
```

### Running the Docker Container

```bash
# Interactive shell
docker run -it --rm brain-image-implementation:latest

# Run a specific script
docker run --rm brain-image-implementation:latest python scripts/gen_embeddings.py

# With GPU support (if available)
docker run --rm --gpus all brain-image-implementation:latest python scripts/train_nice.py

# Mount data directory
docker run --rm -v /path/to/data:/app/data brain-image-implementation:latest python scripts/gen_embeddings.py
```

## Singularity/Apptainer

### Building the Singularity Image

```bash
# Using the build script
./build_singularity.sh

# Or manually with Apptainer
apptainer build brain-image-implementation.sif singularity.def

# Or with Singularity
singularity build brain-image-implementation.sif singularity.def
```

### Running the Singularity Container

```bash
# Interactive shell
apptainer shell brain-image-implementation.sif

# Run a specific script
apptainer exec brain-image-implementation.sif python scripts/gen_embeddings.py

# With GPU support (if available)
apptainer exec --nv brain-image-implementation.sif python scripts/train_nice.py

# Mount data directory
apptainer exec -B /path/to/data:/app/data brain-image-implementation.sif python scripts/gen_embeddings.py
```

## Converting Docker to Singularity

If you have a Docker image, you can convert it to Singularity:

```bash
# Convert Docker image to Singularity
apptainer build brain-image-implementation.sif docker://brain-image-implementation:latest
```

## Cluster Usage

For HPC clusters, the Singularity/Apptainer image is recommended:

1. Build the image on your local machine or a build node
2. Transfer the `.sif` file to the cluster
3. Use in your job scripts:

```bash
#!/bin/bash
#SBATCH --job-name=brain-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Run your script
apptainer exec brain-image-implementation.sif python scripts/train_nice.py
```

## Environment Variables

The container sets the following environment variables:
- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered
- `PYTHONDONTWRITEBYTECODE=1`: Prevents writing .pyc files
- `PATH`: Includes uv in the PATH

## Data Mounting

When running containers, you'll typically need to mount your data directories:

```bash
# Docker
docker run --rm -v /path/to/data:/app/data brain-image-implementation:latest

# Singularity
apptainer exec -B /path/to/data:/app/data brain-image-implementation.sif
```

## Troubleshooting

### Permission Issues
If you encounter permission issues with Singularity, try:
```bash
apptainer build --fakeroot brain-image-implementation.sif singularity.def
```

### GPU Issues
For GPU support, ensure you have the appropriate drivers and runtime:
```bash
# Check if GPU is available
nvidia-smi

# Run with GPU support
apptainer exec --nv brain-image-implementation.sif python -c "import torch; print(torch.cuda.is_available())"
``` 