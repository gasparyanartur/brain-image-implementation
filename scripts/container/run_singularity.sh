#!/bin/bash

image_path=${APPTAINER_IMAGE_PATH}
# if image_path is not set, use the latest image path
if [ -z "$image_path" ]; then
    image_path=$(ls -t images/brain_*.sif | head -n 1)
fi

mount_points=()

if [ -z "$CONTAINER_VERBOSE" ]; then
    SILENCE=1
else
    SILENCE=0
fi


if [ -d "/proj" ]; then
    if [ -z "$SILENCE" ]; then
        echo "Mounting /proj"
    fi
    mount_points+=("--bind /proj/proj")
fi

if [ -d "/home" ]; then
    if [ -z "$SILENCE" ]; then
        echo "Mounting /home"
    fi
    mount_points+=("--bind /home:/home")
fi

if [ -z "$SILENCE" ]; then
    echo "Running singularity image: $image_path"
fi

# Set environment variables for the container
export PROJECT_WORKSPACE_DIR=/workspace
export PYTHONPATH="/workspace/src:$PYTHONPATH"

# Pass through important environment variables
export_env_args=("--env PYTHONUNBUFFERED=1")
if [ -n "$WANDB_API_KEY" ]; then
    export_env_args+=("--env WANDB_API_KEY=$WANDB_API_KEY")
fi

CLI_ARGS="$@"
if [ -z "SILENCE" ]; then
    echo "CLI_ARGS: $CLI_ARGS"
fi

unset PYTHONSTARTUP

apptainer exec \
--nv \
--bind $PWD:/workspace \
--home /workspace \
--workdir /workspace \
--pwd /workspace \
--env PROJECT_WORKSPACE_DIR=/workspace \
${export_env_args[*]} \
${mount_points[*]} \
$image_path \
$CLI_ARGS

if [ $? -eq 0 ]; then
    echo "Command $CLI_ARGS completed at $(date) with exit code 0"
else
    echo "Command $CLI_ARGS failed at $(date) with exit code $?"
    exit 1
fi