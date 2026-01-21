#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=berzelius-cpu
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/prepare_data/%j.out
#SBATCH --account=Berzelius-2025-278


echo "Job ID: $SLURM_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"

image_path=${APPTAINER_IMAGE_PATH}
# if image_path is not set, use the latest image path
if [ -z "$image_path" ]; then
    image_path=$(ls -t images/brain_*.sif | head -n 1)
fi

dataset=$1
echo "Dataset: $dataset"

cli_args=("${@:2}")
echo "cli_args: ${cli_args[@]}"

# If data type is not given, we manually set
if [ $dataset == "things-eeg2" ]; then
    has_img_flag=0
    for ((i=0; i < ${#cli_args[@]}; i++)); do
        arg=${cli_args[$i]}
        value=${cli_args[$i+1]}
        
        if [ [ $arg == "--download_types" ] || [ $arg == "-t" ] ] && [ $value == "imgs" ]; then
           has_img_flag=1
           break
        fi
    done

    if [ $has_img_flag -eq 0 ]; then
       cli_args+=("--download_types" "imgs")
    fi

elif [ $dataset == "alljoined-16m" ]; then
    has_img_flag=0
    for ((i=0; i < ${#cli_args[@]}; i++)); do
        arg=${cli_args[$i]}
        value=${cli_args[$i+1]}
        
        if [ [ $arg == "--download_types" ] || [ $arg == "-t" ] ] && [ $value == "stim" ]; then
           has_img_flag=1
           break
        fi
    done

    if [ $has_img_flag -eq 0 ]; then
       cli_args+=("--download_types" "stim")
    fi
fi

echo ${cli_args[@]}
exit 1

cmd="/workspace/scripts/data/${dataset}/prepare.sh ${cli_args[@]}"
echo "Preparing data with command: $cmd"

./scripts/container/run_singularity.sh $cmd

if [ $? -eq 0 ]; then
    echo "Prepared data successfully"
else
    echo "Data preparation failed with exit code $?"
    exit 1
fi

echo "Job completed at $(date)" 