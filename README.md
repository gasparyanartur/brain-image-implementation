# Brain Image Implementation

## Project Description

(TODO: Update this)

## Environments

Scripts in this repo run either locally with `uv` or on a SLURM cluster via `ssub`. Both forms are shown throughout this README where relevant.

**Local:** Activate your venv and run scripts directly (see [Python environment](#3-python-environment)):

```bash
source .venv/bin/activate
python scripts/training/train_eeg.py --config-name=train_eeg_align
```

**Cluster:** Use `ssub` to submit any command as a SLURM job wrapped in the Singularity container (see [SLURM job submission](#slurm-job-submission-ssub)):

```bash
ssub train_eeg python scripts/training/train_eeg.py --config-name=train_eeg_align
```

## Setup

### 1. Directory structure

The project expects several large-data directories to exist at the repo root. If you are working on a cluster, symlink these to your storage volume — they will hold datasets, cached tensors, model weights, and logs that are too large to keep in the repo.

```bash
ln -s /path/to/storage/data        data
ln -s /path/to/storage/tensorcache tensorcache
ln -s /path/to/storage/logs        logs
ln -s /path/to/storage/models      models
ln -s /path/to/storage/.cache      .cache
ln -s /path/to/storage/experiments experiments
ln -s /path/to/storage/.cache/huggingface/hub ~/.cache/huggingface/
mkdir -p logs/slurm
```

If you are working locally without a separate storage volume, you can just create the directories directly with `mkdir`.

### 2. Environment variables

The project loads credentials and cluster settings from a `.env` file at the repo root. Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Required keys:

```bash
WANDB_API_KEY=...           # from https://wandb.ai/authorize
HF_API_TOKEN=...            # from https://huggingface.co/settings/tokens

# Cluster (SLURM + Singularity)
SBATCH_ACCOUNT=...          # your SLURM allocation
SBATCH_GPU_PARTITION=...    # GPU partition name
SBATCH_CPU_PARTITION=...    # CPU partition name
STORAGE_DIR=...             # path to your storage volume (mounted into container)
```

The `.env` file is automatically sourced by `ssub` before job submission, so all keys are available inside the Singularity container at runtime.

### 3. Python environment

**Local (UV):**

UV is the recommended local environment manager. Install it and set up the project:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv sync
```

Try importing `torch` in a Python shell to confirm the installation.

**SLURM (Singularity):**

On the cluster, the project runs inside a Singularity/Apptainer container. Utility scripts for building and running images are in `scripts/container/`.

Build the base image first — this installs CUDA, Python 3.12, and PyTorch, and takes a while. It requires `sudo`, so it is best done on a local machine and then rsynced to the cluster:

```bash
DEFINITION_FILE=scripts/container/singularity_base.def \
IMAGE_FILE=images/singularity_base.sif \
./scripts/container/build_singularity.sh
```

Once the base image is in place, build the main project image:

```bash
./scripts/container/build_singularity.sh
```

The output will be an image named `images/brain_<datetime>.sif`. This is the image used for all SLURM jobs. The most recently built image is picked up automatically by `run_singularity.sh`.

### 4. Verify setup

Before running any training, verify that the container can access the GPU and connect to external services.

Test CUDA access:

```bash
SBATCH_GROUP=gpu-light ssub test_cuda python scripts/utils/test_cuda.py
```

Test wandb connectivity:

```bash
SBATCH_GROUP=cpu-light ssub test_wandb python scripts/utils/test_wandb.py
```

Both use the `devel` QOS for fast queue access. Check the logs under `logs/slurm/` for results.

### 5. Data

Each dataset has its own `prepare.sh` under `scripts/data/<dataset>/` that downloads and preprocesses the data. Pass `--modality eeg` (default) for EEG recordings or `--modality img` for stimulus images. Subject-level EEG jobs accept `-s <sub>` and are designed to run in parallel as SLURM array jobs using `$SLURM_ARRAY_TASK_ID`.

**Things-EEG2**

EEG — subs 1–10, one task per subject:

```bash
# Local
for s in $(seq 1 10); do bash scripts/data/things-eeg2/prepare.sh -s $s; done
# SLURM (array job)
SBATCH_GROUP=cpu SBATCH_ARRAY=1-10 ssub prepare_things_eeg \
  bash scripts/data/things-eeg2/prepare.sh -s '$SLURM_ARRAY_TASK_ID'
```

Images — downloaded once, no subject argument:

```bash
# Local
bash scripts/data/things-eeg2/prepare.sh --modality img
# SLURM
SBATCH_GROUP=cpu ssub prepare_things_eeg_img bash scripts/data/things-eeg2/prepare.sh --modality img
```

**AllJoined-16M**

EEG — subs 1–20, one task per subject:

```bash
# Local
for s in $(seq 1 20); do bash scripts/data/alljoined-16m/prepare.sh -s $s; done
# SLURM (array job)
SBATCH_GROUP=cpu SBATCH_ARRAY=1-20 ssub prepare_alljoined \
  bash scripts/data/alljoined-16m/prepare.sh -s '$SLURM_ARRAY_TASK_ID'
```

Stimuli — downloaded once, no subject argument:

```bash
# Local
bash scripts/data/alljoined-16m/prepare.sh --modality img
# SLURM
SBATCH_GROUP=cpu ssub prepare_alljoined_stim bash scripts/data/alljoined-16m/prepare.sh --modality img
```

> **Note:** Single-quoting `'$SLURM_ARRAY_TASK_ID'` in the ssub command prevents the variable from being expanded at submission time; SLURM expands it at runtime inside each array task.

### 6. Generate image embeddings

The training loop assumes all image latents are precomputed and cached in `tensorcache/`. This step encodes the stimulus images with the image encoders used during training. A full list of supported encoders is in `src/brain_image/model/encoder/img_encoder/union.py`.

```bash
# Local (all encoders):
python scripts/data/generate_embeddings.py
# or for specific encoders only:
python scripts/data/generate_embeddings.py model_names=[clip_vith14]

# SLURM (all encoders):
SBATCH_GROUP=gpu ssub generate_embeddings python scripts/data/generate_embeddings.py
```

The default set of encoders is configured in `src/brain_image/configs/generate_embeddings.yaml`.

### 7. Generate text captions

Text captions are generated locally using a Qwen VL model. Run this before generating text embeddings:

```bash
# Local:
python scripts/data/generate_text_captions_local.py --config-name=generate_text_captions_local

# SLURM:
SBATCH_GROUP=gpu ssub generate_captions python scripts/data/generate_text_captions_local.py --config-name=generate_text_captions_local
```

Captions are written to `data/things-eeg2/captions/local.jsonl` (one JSON line per image with `path` and `caption` fields). Already-captioned images are skipped on re-runs. The model name and other settings are in `src/brain_image/configs/generate_text_captions_local.yaml`.

### 8. Generate text embeddings

Once captions exist, encode them with one or more text encoders and cache the results in `tensorcache/`. Supported encoders: `t5_base`, `t5_large`, `clip_vitl14_text`, `clip_vitb32_text`, `llama3_8b`, `gemma_embedding_300m`.

```bash
# Local (all encoders in the config):
python scripts/data/generate_text_embeddings.py --config-name=generate_text_embeddings
# or for specific encoders only:
python scripts/data/generate_text_embeddings.py model_names=[t5_base]

# SLURM:
SBATCH_GROUP=gpu ssub generate_text_embeddings python scripts/data/generate_text_embeddings.py --config-name=generate_text_embeddings
```

The default encoder list and caption path are in `src/brain_image/configs/generate_text_embeddings.yaml`.

### 9. Generate statistics

After embeddings are cached, compute per-split mean and standard deviation statistics for both the EEG signals and all embeddings (image and/or text). These stats are saved to `statistics/datasets/<dataset>/<split>/` and are used for normalisation during training.

**Image embedding stats:**

```bash
python scripts/data/generate_stats.py
# or for specific encoders:
python scripts/data/generate_stats.py model_names=[clip_vith14]
# or on SLURM:
SBATCH_GROUP=cpu ssub generate_stats python scripts/data/generate_stats.py
```

**Text embedding stats:**

```bash
python scripts/data/generate_stats.py --config-name=generate_text_stats
# or for specific encoders:
python scripts/data/generate_stats.py --config-name=generate_text_stats model_names=[t5_base]
```

The default dataset and encoder list for each are configured in `src/brain_image/configs/generate_stats.yaml` and `src/brain_image/configs/generate_text_stats.yaml` respectively.

### 10. Train

```bash
python scripts/training/train_eeg.py --config-name=train_eeg_align
# or on SLURM:
ssub train_eeg python scripts/training/train_eeg.py --config-name=train_eeg_align
```

For text-encoder alignment, use the `train_eeg_align_text` config:

```bash
python scripts/training/train_eeg.py --config-name=train_eeg_align_text
```

### 11. Evaluate a trained run

After training, evaluate a checkpoint with `scripts/evaluation/test_eeg.py`. Pass the run directory (the `logs/train/<run_name>` folder produced by the logger) and it will find the best checkpoint automatically:

```bash
python scripts/evaluation/test_eeg.py logs/train/my_run
# or on SLURM:
ssub test_eeg python scripts/evaluation/test_eeg.py logs/train/my_run
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint_selection` | `min` | How to pick checkpoint: `min`, `max`, or `last` |
| `--checkpoint_metric` | `val-loss` | Metric used to select the checkpoint |
| `--output_dir` | `<run>/test/` | Where to write metrics and reconstructed images |
| `--metrics` | all | Subset of metrics to compute |

Results are written as `test_metrics.json` and reconstructed images inside the output directory.

### 12. Run a single experiment locally

To train and immediately evaluate one parameter combination from a param file without SLURM, use `launch_local.sh`:

```bash
./scripts/evaluation/launch_local.sh <task_id> <param_path> <config_name> <train_script> <test_script>
```

For example, to run the third configuration (0-based) from the text-vs-image encoder sweep:

```bash
./scripts/evaluation/launch_local.sh 2 \
    scripts/params/text_vs_img_encoders.json \
    train_eeg_align_text \
    scripts/training/train_eeg.py \
    scripts/evaluation/test_eeg.py
```

To list all parameter combinations in a file and their indices:

```bash
for i in $(seq 0 $(( $(python scripts/slurm/param_parser.py scripts/params/text_vs_img_encoders.json -s) - 1 ))); do
    echo "$i: $(python scripts/slurm/param_parser.py scripts/params/text_vs_img_encoders.json -i $i)"
done
```

Results land in `experiments/local/<timestamp>_task<id>/`.

### 13. Sweep experiments

To run many training jobs with different hyperparameters, define a **param file** in `scripts/slurm/params/` and launch the full pipeline with `run_experiment_pipeline.sh`.

**Param file format** — `scripts/slurm/params/<name>.json`:

```json
{
    "entries": [
        {
            "keys": ["model.align_img_encoder"],
            "values": [["clip_vitl14", "clip_vith14", "synclr_vitb16"]]
        },
        {
            "keys": ["model.eeg_encoder"],
            "values": [["nice", "atms"]]
        }
    ]
}
```

Each entry is one sweep axis. Multiple entries are crossed as a cartesian product — the example above produces `3 × 2 = 6` combinations. To keep keys **paired** (i.e. move together instead of crossing), list them in the same entry:

```json
{
    "entries": [
        {
            "keys": ["model.lr", "model.eeg_encoder"],
            "values": [["1e-4", "1e-3"], ["atms", "nice"]]
        }
    ]
}
```

This produces only `(1e-4, atms)` and `(1e-3, nice)` — not all four combinations.

**Run the pipeline:**

```bash
./scripts/evaluation/run_experiment_pipeline.sh \
    <experiment_name> <param_path> <config_name> \
    <train_script> <test_script> \
    [cli_args...]
```

For example:

```bash
TEST_HPARAMS="model.align_img_encoder model.eeg_encoder" \
  ./scripts/evaluation/run_experiment_pipeline.sh \
    encoders \
    scripts/slurm/params/encoders.json \
    train_eeg_align \
    scripts/training/train_eeg.py \
    scripts/evaluation/test_eeg.py
```

This submits two chained SLURM jobs automatically:

1. **Array (train + test)** — one SLURM array task per parameter combination. Each task trains its configuration then immediately evaluates the resulting run. Tasks use `--requeue` so they restart automatically on node failure or preemption. The train and test scripts are passed explicitly, making the pipeline reusable for different model types.
2. **Aggregate** — collects `test_metrics.json` from every run in the experiment directory once all array tasks finish, and writes a single `experiments/<name>/aggregated_metrics.csv`.

Set `TEST_HPARAMS` to a space-separated list of dotted config keys (e.g. `"model.lr model.eeg_encoder"`) to include those hyperparameter values as columns in the CSV.

**Aggregate metrics standalone** — if you only need to re-aggregate results from an existing experiment:

```bash
python scripts/evaluation/aggregate_metrics.py \
    --experiment_dir experiments/encoders \
    --hparams model.align_img_encoder model.eeg_encoder
```

Add `-t` to tail logs immediately after submission:

```bash
SBATCH_GROUP=gpu-light ssub train_eeg -t python scripts/training/train_eeg.py --config-name=train_eeg_align model.max_epochs=50
```

## SLURM job submission (`ssub`)

`scripts/slurm/ssub.sh` is the main entrypoint for submitting jobs to SLURM. It automatically wraps your command in the Singularity container and handles log routing, resource defaults, and environment forwarding.

```
Usage: ssub <job_name> [-t] [--dry-run] <command...>
```

- `job_name` is used as the SLURM job name and as the log subdirectory under `logs/slurm/`.
- `-t` tails the job log immediately after submission.
- `--dry-run` prints the final `sbatch` command without submitting.

**Resource groups** (`SBATCH_GROUP`):

Defaults are loaded from [`scripts/slurm/ssub_groups.conf`](scripts/slurm/ssub_groups.conf) — edit that file to add or modify groups.

| Group       | CPUs | Mem   | Time       | GPUs | QOS     |
|-------------|------|-------|------------|------|---------|
| `gpu`       | 32   | 128G  | 1-00:00:00 | 1    |         |
| `gpu-light` | 8    | 32G   | 01:00:00   | 1    | `devel` |
| `cpu`       | 32   | 128G  | 1-00:00:00 | —    |         |
| `cpu-light` | 8    | 32G   | 01:00:00   | —    | `devel` |

The `*-light` groups use the `devel` QOS for fast queue access and are useful for quick tests. Any resource can be overridden with `SBATCH_*` env vars:

```bash
SBATCH_GROUP=gpu-light SBATCH_TIME="02:00:00" ssub my_job python my_script.py
```

For free-form sbatch flag overrides (e.g. constraints, QOS):

```bash
SBATCH_OVERRIDE="--qos=debug --constraint=foo" ssub my_job python my_script.py
```

## Configs

Configs live in `src/brain_image/configs/` and use [Hydra](https://hydra.cc/docs/intro/). The hierarchy mirrors the model structure: there are groups for the dataset, model, trainer, encoder, and augmentation, all composed into a top-level training config.

Each training script has a default config file, specified in its `@hydra.main` decorator. Override it on the CLI with `--config-name`:

```bash
python scripts/training/train_eeg.py --config-name=train_eeg_align
```

To swap out an entire config group (e.g. use a different encoder), use the group override syntax:

```bash
[group]@[target_variable]=[option]
```

For example, to load the `atms` encoder config into `model.eeg_encoder`:

```bash
python scripts/training/train_eeg.py eeg_encoder@model.eeg_encoder=atms
```

To override a single scalar value:

```bash
python scripts/training/train_eeg.py model.eeg_encoder.num_layers=6
```

When a child config inherits from a parent that already sets a config group, use `override` to replace it rather than append:

```yaml
defaults:
  - train_eeg
  - override augmentation: eeg
```

The schema for each config group is defined as a dataclass in `src/brain_image/configs.py` (e.g. `EEGEncoderConfig`, `TrainerConfig`). This is the authoritative reference for what parameters are available and what their types and defaults are.

