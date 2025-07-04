#!/bin/bash

image_path=${APPTAINER_IMAGE_PATH:-/home/x_artga/projdir/images/brain_2025_07_03.sif}
dest_dir=${APPTAINER_IMAGE_DEST_DIR:-/home/x_artga/projdir/images}

image_name=$(basename $image_path)
final_image_path="${dest_dir}/${image_name}"

rsync -avz --progress --partial $image_path $final_image_path