#!/bin/bash

image_path=${APPTAINER_IMAGE_PATH}
# if image_path is not set, use the latest image path
if [ -z "$image_path" ]; then
    image_path=$(ls -t images/brain_*.sif | head -n 1)
fi

mount_points=()

if [ -d "/proj" ]; then
    echo "Mounting /proj"
    mount_points+=("--bind /proj/proj")
fi

if [ -d "/home" ]; then
    echo "Mounting /home"
    mount_points+=("--bind /home:/home")
fi

echo "Running singularity image: $image_path"

# Set environment variables for the container
export PROJECT_WORKSPACE_DIR=/workspace
export PYTHONPATH="/workspace/src:$PYTHONPATH"

# Pass through important environment variables
export_env_args=("--env PYTHONUNBUFFERED=1")
if [ -n "$WANDB_API_KEY" ]; then
    export_env_args+=("--env WANDB_API_KEY=$WANDB_API_KEY")
fi

CLI_ARGS="$@"
echo "CLI_ARGS: $CLI_ARGS"

apptainer run \
--nv \
--bind $PWD:/workspace \
--home /workspace \
--workdir /workspace \
--pwd /workspace \
--env PROJECT_WORKSPACE_DIR=/workspace \
${export_env_args[*]} \
${mount_point[*]} \
$image_path $CLI_ARGS