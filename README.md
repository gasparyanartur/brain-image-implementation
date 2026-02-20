# Brain Image Implementation

## Project Description

(TODO: Update this)

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

Download and prepare the dataset. The example below uses subject 8 of Things-EEG2; repeat for any subjects you want to include:

```bash
python scripts/setup_data.py --subs 8
# or on SLURM:
ssub setup_data python scripts/setup_data.py --subs 8
```

### 6. Generate embeddings

The training loop assumes all image latents are precomputed and cached in `tensorcache/`. This step encodes the stimulus images with the image encoders used during training. A full list of supported encoders is in `src/brain_image/model/img_encoder.py`.

```bash
python scripts/generate_embeddings.py
# or for specific encoders only:
python scripts/generate_embeddings.py model_names=[clip_vith14]
```

The default set of encoders to generate is configured in `src/brain_image/configs/generate_embeddings.yaml`.

### 7. Train

```bash
python scripts/training/train_eeg.py --config-name=train_eeg_align
# or on SLURM:
ssub train_eeg python scripts/training/train_eeg.py --config-name=train_eeg_align
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

| Group       | CPUs | Mem   | Time       | GPUs | QOS   |
|-------------|------|-------|------------|------|-------|
| `gpu`       | 32   | 128G  | 1-00:00:00 | 1    |       |
| `gpu-light` | 8    | 32G   | 01:00:00   | 1    | devel |
| `cpu`       | 32   | 128G  | 1-00:00:00 | —    |       |
| `cpu-light` | 8    | 32G   | 01:00:00   | —    | devel |

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


### 2. Environment variables

Copy `.env.example` to `.env` and fill in your credentials:

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

### 3. Python environment

**Local (UV):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv sync
```

**SLURM (Singularity):**

Build the base image first (requires sudo, best done locally then rsynced):

```bash
DEFINITION_FILE=scripts/container/singularity_base.def \
IMAGE_FILE=images/singularity_base.sif \
./scripts/container/build_singularity.sh
```

Then build the main image:

```bash
./scripts/container/build_singularity.sh
```

Output will be `images/brain_<datetime>.sif`.

### 4. Verify setup

Test CUDA access:

```bash
SBATCH_GROUP=gpu-light ssub test_cuda python scripts/utils/test_cuda.py
```

Test wandb connectivity:

```bash
SBATCH_GROUP=cpu-light ssub test_wandb python scripts/utils/test_wandb.py
```

### 5. Data

```bash
python scripts/setup_data.py --subs 8
# or on SLURM:
ssub setup_data python scripts/setup_data.py --subs 8
```

### 6. Generate embeddings

Precompute image latents into `tensorcache/` before training:

```bash
python scripts/generate_embeddings.py
# specific encoders:
python scripts/generate_embeddings.py model_names=[clip_vith14]
```

### 7. Train

```bash
python scripts/training/train_eeg.py --config-name=train_eeg_align
# or on SLURM:
ssub train_eeg python scripts/training/train_eeg.py --config-name=train_eeg_align
```

Add `-t` to tail logs immediately after submission:

```bash
SBATCH_GROUP=gpu-light ssub train_eeg -t python scripts/training/train_eeg.py --config-name=train_eeg_align model.max_epochs=50
```

## SLURM job submission (`ssub`)

`scripts/slurm/ssub.sh` is the main entrypoint for submitting jobs. It wraps your command in the Singularity container automatically.

```
Usage: ssub <job_name> [-t] [--dry-run] <command...>
```

**Resource groups** (`SBATCH_GROUP`):

| Group       | CPUs | Mem   | Time       | GPUs | QOS   |
|-------------|------|-------|------------|------|-------|
| `gpu`       | 32   | 128G  | 1-00:00:00 | 1    |       |
| `gpu-light` | 8    | 32G   | 01:00:00   | 1    | devel |
| `cpu`       | 32   | 128G  | 1-00:00:00 | —    |       |
| `cpu-light` | 8    | 32G   | 01:00:00   | —    | devel |

Any resource can be overridden with `SBATCH_*` env vars:

```bash
SBATCH_GROUP=gpu-light SBATCH_TIME="02:00:00" ssub my_job python my_script.py
```

## Configs

Configs live in `src/brain_image/configs/` and use [Hydra](https://hydra.cc/docs/intro/).

Each training script has a default config file, found in its `@hydra.main` decorator. Override it with `--config-name`:

```bash
python scripts/training/train_eeg.py --config-name=train_eeg_align
```

Every parameter can be overridden on the CLI. To swap a config group, use:

```bash
[group]@[target_variable]=[option]
```

For example, to load the `atms` encoder config into `model.eeg_encoder`:

```bash
python scripts/training/train_eeg.py eeg_encoder@model.eeg_encoder=atms
```

To override a scalar value:

```bash
python scripts/training/train_eeg.py model.eeg_encoder.num_layers=6
```

When a child config inherits from a parent that already sets a config group, use `override` to replace it:

```yaml
defaults:
  - train_eeg
  - override augmentation: eeg
```

The values of each config can be traced through the YAML files and their corresponding dataclasses (e.g. `EEGEncoderConfig` in `src/brain_image/configs.py`).

