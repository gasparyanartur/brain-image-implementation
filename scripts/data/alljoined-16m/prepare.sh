#!/usr/bin/env bash
set -euo pipefail

# We define a dispatch system, to allow overriding the default arguments

download_kwargs=()
preprocess_kwargs=()

declare -A download_member=(
    ["-d"]=1
    ["--data_path"]=1
    ["--subs"]=1
    ["-s"]=1
    ["--raw_eeg_dir"]=1
    ["--stim_dir"]=1
    ["-t"]=1
    ["--download_types"]=1
)

declare -A preprocess_member=(
    ["-d"]=1
    ["--data_path"]=1
    ["-s"]=1
    ["--sub"]=1
    ["--preprocessed_eeg_dir"]=1
    ["--n_ses"]=1
    ["--sfreq"]=1
    ["--mvnn_dim"]=1
)

args=("$@")
modality="eeg"
has_download_types=0

i=0
N=${#args[@]}

is_keyword() {
  local k="$1"
  [[ -v download_member["$k"] || -v preprocess_member["$k"] || "$k" == "--modality" ]]
}

while (( i < N )); do
  key="${args[i]}"

  if ! is_keyword "$key"; then
    echo "Unexpected token: $key"
    exit 1
  fi

  # collect values until next keyword
  values=()
  ((++i))

  while (( i < N )) && ! is_keyword "${args[i]}"; do
    values+=("${args[i]}")
    ((++i))
  done

  # dispatch
  if [[ "$key" == "--modality" ]]; then
    if (( ${#values[@]} == 0 )); then
      echo "Missing value for --modality (eeg|img)"
      exit 2
    fi
    modality="${values[0]}"
    continue
  fi

  if [[ "$key" == "-t" || "$key" == "--download_types" ]]; then
    has_download_types=1
  fi

  if [[ -v download_member["$key"] ]]; then
    download_kwargs+=("$key" "${values[@]}")
  fi

  if [[ -v preprocess_member["$key"] ]]; then
    preprocess_kwargs+=("$key" "${values[@]}")
  fi
done

if (( has_download_types == 0 )); then
  case "$modality" in
    eeg) download_kwargs+=("--download_types" "eeg" "stim-order") ;;
    img) download_kwargs+=("--download_types" "stim") ;;
    *)
      echo "Invalid --modality: $modality (expected eeg|img)"
      exit 2
      ;;
  esac
fi

download_cmd_path=$(dirname "$0")/download.py
echo "Running python ${download_cmd_path} ${download_kwargs[@]}"
python ${download_cmd_path} "${download_kwargs[@]}"

preprocess_cmd_path=$(dirname "$0")/preprocess.py
echo "Running python ${preprocess_cmd_path} ${preprocess_kwargs[@]}"
python ${preprocess_cmd_path} "${preprocess_kwargs[@]}"
