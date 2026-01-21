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
    ["--img_dir"]=1
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

i=0
N=${#args[@]}

is_keyword() {
  local k="$1"
  [[ -v download_member["$k"] || -v preprocess_member["$k"] ]]
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
  if [[ -v download_member["$key"] ]]; then
    download_kwargs+=("$key" "${values[@]}")
  fi

  if [[ -v preprocess_member["$key"] ]]; then
    preprocess_kwargs+=("$key" "${values[@]}")
  fi
done


download_cmd_path=$(dirname "$0")/download.py
echo "Running python ${download_cmd_path} ${download_kwargs[@]}"
python ${download_cmd_path} "${download_kwargs[@]}"

preprocess_cmd_path=$(dirname "$0")/preprocess.py
echo "Running python ${preprocess_cmd_path} ${preprocess_kwargs[@]}"
python ${preprocess_cmd_path} "${preprocess_kwargs[@]}"