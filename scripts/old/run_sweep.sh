#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/sweep/%A_%a.out
#SBATCH --account=Berzelius-2025-278

echo "Job ID: $SLURM_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"

task_id=$SLURM_ARRAY_TASK_ID

image_path=${APPTAINER_IMAGE_PATH}
# if image_path is not set, use the latest image path
if [ -z "$image_path" ]; then
    image_path=$(ls -t images/brain_*.sif | head -n 1)
fi

param_path="$1"
CLI_ARGS="${@:2}"

echo "PARAM_PATH: $param_path"
echo "CLI_ARGS: $CLI_ARGS"

export CONTAINER_VERBOSE=$CONTAINER_VERBOSE

if [ -z "$param_path" ]; then
    echo "Missing param_path"
    array_params=""
    
else
    array_params_count=$(
        ./scripts/container/run_singularity.sh python scripts/slurm/param_parser.py $param_path -s
    )
    echo "Array Params Count: $array_params_count"

    if [[ $array_param_count -gt $SLURM_ARRAY_TASK_MAX ]]; then
        echo "Array parameter count $array_param_count is greater than SLURM_ARRAY_TASK_MAX $SLURM_ARRAY_TASK_MAX - exiting"
        exit 1
    fi

    array_params=$(
       ./scripts/container/run_singularity.sh python scripts/slurm/param_parser.py $param_path  -i $task_id
    )
    if [[ -z ${array_params} ]]; then 
        echo "No array parameters found for task $task_id - exiting"
        exit 1
    fi
fi

args="$CLI_ARGS $array_params"
echo "Program Arguments: $args"

./scripts/container/run_singularity.sh $args