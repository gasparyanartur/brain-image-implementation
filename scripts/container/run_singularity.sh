#!/bin/bash

image_path=${APPTAINER_IMAGE_PATH:-/home/x_artga/projdir/images/brain_2025_07_04_10_44_01.sif}

apptainer shell \
--nv \
--home /workspace \
--bind $PWD:/workspace \
--workdir /workspace \
--pwd /workspace \
$image_path "$@"