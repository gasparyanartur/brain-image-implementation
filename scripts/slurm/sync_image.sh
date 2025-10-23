#!/bin/bash

image_path=${APPTAINER_IMAGE}
if [ -z "$image_path" ]; then
    # Find the latest image in the images directory
    image_path=$(ls -t images/brain_*.sif | head -n 1)
    if [ -z "$image_path" ]; then
        echo "No image found in images directory"
        exit 1
    fi
fi

dest_dir=${APPTAINER_IMAGE_DEST_DIR:-/home/x_artga/projdir/images}

image_name=$(basename $image_path)
cluster_url=${CLUSTER_URL:-x_artga@berzelius.nsc.liu.se}
final_image_path="${cluster_url}:${dest_dir}/${image_name}"

echo "Syncing image $image_path to $final_image_path"
rsync -avzh --progress --partial $image_path $final_image_path