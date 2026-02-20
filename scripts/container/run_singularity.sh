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


if [ -n "$STORAGE_DIR" ] && [ -d "$STORAGE_DIR" ]; then
    if [ -z "$SILENCE" ]; then
        echo "Mounting $STORAGE_DIR"
    fi
    mount_points+=("--bind $STORAGE_DIR")
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
export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Pass through all current environment variables
export_env_args=()
while IFS='=' read -r key value; do
    export_env_args+=("--env" "${key}=${value}")
done < <(env)

CLI_ARGS="$@"
if [ -z "$SILENCE" ]; then
    echo "CLI_ARGS: $CLI_ARGS"
fi

unset PYTHONSTARTUP

apptainer exec \
--nv \
--bind $PWD:/workspace \
--home /workspace \
--workdir /workspace \
--pwd /workspace \
"${export_env_args[@]}" \
${mount_points[*]} \
$image_path \
$CLI_ARGS

if [ $? -eq 0 ]; then
    echo "Command $CLI_ARGS completed at $(date) with exit code 0"
else
    echo "Command $CLI_ARGS failed at $(date) with exit code $?"
    exit 1
fi