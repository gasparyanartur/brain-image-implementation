#!/bin/bash

image_path=${APPTAINER_IMAGE_PATH}
# if image_path is not set, use the latest image path
if [ -z "$image_path" ]; then
    image_path=$(ls -t images/brain_*.sif | head -n 1)
fi

echo "Running singularity image: $image_path"
apptainer run \
--nv \
--home /workspace \
--bind $PWD:/workspace \
--bind /proj:/proj \
--bind /home:/home \
--workdir /workspace \
--pwd /workspace \
$image_path "$@"