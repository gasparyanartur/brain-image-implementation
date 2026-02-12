#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=berzelius-cpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/prepare_data/%j.out
#SBATCH --account=Berzelius-2025-278

dataset=$1
if [[ -z "${dataset:-}" ]]; then
    echo "Usage: $0 <dataset> [--modality eeg|img] [prepare.sh args...]"
    exit 2
fi
echo "Dataset: $dataset"

cli_args=("${@:2}")

modality="eeg"
filtered_args=()
for ((i=0; i<${#cli_args[@]}; i++)); do
    case "${cli_args[$i]}" in
        --modality)
            if (( i + 1 >= ${#cli_args[@]} )); then
                echo "Error: --modality requires a value (eeg|img)"
                exit 2
            fi
            modality="${cli_args[$((i+1))]}"
            i=$((i+1))
            ;;
        *)
            filtered_args+=("${cli_args[$i]}")
            ;;
    esac
done
cli_args=("${filtered_args[@]}")
echo "cli_args: ${cli_args[@]}"

has_sub=0
has_download_types=0
for ((i=0; i<${#cli_args[@]}; i++)); do
    case "${cli_args[$i]}" in
        -s|--sub) has_sub=1 ;;
        -t|--download_types) has_download_types=1 ;;
    esac
done

if (( has_sub == 0 )) && [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && [[ "$modality" == "eeg" ]]; then
    echo "Sub not defined, setting to array index $SLURM_ARRAY_TASK_ID"
    cli_args+=(-s "$SLURM_ARRAY_TASK_ID")
fi

cmd="/workspace/scripts/data/${dataset}/prepare.sh --modality ${modality} ${cli_args[@]}"
echo "Preparing data with command: $cmd"

./scripts/container/run_singularity.sh $cmd
