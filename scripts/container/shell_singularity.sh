#!/bin/bash

image_path=${APPTAINER_IMAGE_PATH}
# if image_path is not set, use the latest image path
if [ -z "$image_path" ]; then
    image_path=$(ls -t images/brain_*.sif | head -n 1)
fi

mount_points=()

if [ -d "/proj" ]; then
    echo "Mounting /proj"
    mount_points+=("/proj")
fi

if [ -d "/home" ]; then
    echo "Mounting /home"
    mount_points+=("/home")
fi

echo "Launching shell in singularity image: $image_path"
apptainer shell \
--nv \
--bind $PWD:/workspace \
--home /workspace \
--workdir /workspace \
--pwd /workspace \
$(
    for mount_point in "${mount_points[@]}"; do
        echo "--bind $mount_point:$mount_point"
    done
) \
$image_path "$@"