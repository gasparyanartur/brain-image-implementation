#!/bin/bash


src_path=$1
dst_path=$2

cluster_url=${CLUSTER_URL:-x_artga@berzelius.nsc.liu.se}

if [ -z "$src_path" ]; then
    echo "Missing src_path"
    echo "Usage: $0 src_path"
    exit 1
fi

if [ -z "$dst_path" ]; then
    dst_path=$src_path
    echo "Using default dst_path: $dst_path"
fi

final_src_path="${cluster_url}:${src_path}"

echo "Syncing $final_src_path to $dst_path"
rsync -avzh --progress --partial ${final_src_path} ${dst_path}