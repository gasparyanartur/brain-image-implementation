# Brain Image Implementation

## Project Description

Collection of code for training and evaluation of EEG-image models. The primary workflow investigates alignment of EEG signals to image latents, and reconstruction of images from EEG using a diffusion prior. 


# Project Structure

We use Pytorch Lightning for training, and Hydra for configuration. Evaluation code is separate from training, and is designed to be run on trained checkpoints after the fact. There are also various utility scripts for data processing, embedding generation, and SLURM job submission.

In this project, there have been multiple sub-projects:

**EEG-Image Alignment**: Our main track, which builds on Nona's work, [Human-Aligned Image Models Improve Visual Decoding from the Brain](https://github.com/NonaRjb/AlignVis). Here, we take the processed EEG signals, embed them using an EEG encoder, and align the latents to an image-embedding using contrastive loss. The code for this part can be found in `src/brain_image/model/eeg_alignment.py`. Notably, the `do_align` flag controls whether the contrastive loss is applied during training. See `src/brain_image/configs/train_eeg_align.yaml` for sensible default settings for this track.

**Image Reconstruction from EEG**: Here, we implement and train a diffusion prior to map EEG latents to CLIP image space, and then use a pretrained diffusion model to recreate images from them. The `do_recon` flag determines whether we do this part or not. See `src/brain_image/configs/train_eeg_prior.yaml` for default settings for this track. You can also combine both the alignment and reconstruction by setting both flags to true, and using `train_eeg_alignprior.yaml` for default settings.

We have two pipelines: The one from [Reconstructing the Mind's Eye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD) (*Dalle-2 Diffusion Prior paried with SD2.1 + Image Variation*), and the one from [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://github.com/dongyangli-del/eeg_image_decode) (*A simple Diffusion Prior paired with SDXL + IP-adapter*). The Guided EEG pipeline is the one that we have been maintaining. The Mind's Eye pipeline is also there, but is currently broken and needs to be fixed prior to usage. Ideally, there would be a shared interface for them, so that we can easily switch between them and compare results. 

**CoMM Model**: We tried generating a CLIP-latent using our diffusion prior, and jointly encoding that with EEG to get a multimodal representation. We then tried to decode the images using this representation, but it didn't work better than just using the EEG latents, so we have not focused on this track. This part of the code should mostly be working. See `src/brain_image/model/comm_alignment.py` for the model code, and `src/brain_image/configs/train_comm.yaml` for the training config. 

**Text Alignment**: We tried generating text from the dataset using QWEN and aligning the EEG latents to that instead of the image latents. This didn't work much better than the image alignment. The code currently works as a simple plug-in to the alignment pipeline. See `src/brain_image/configs/train_eeg_align_text.yaml` for the training config.

**Low Level Pipeline**: We implement a low-level pipeline which takes the EEG latent and trues to directly reconstruct a blurry image. This is then passed through the frozen Stable Diffusion VAE-decoder to get an initial image for the img-to-img. This code is currently out of date, and needs to be fixed and cleaned up. Especially the training script is completely incompatible with the current codebase, and needs to be rewritten. See `src/brain_image/model/low_level.py` for the model code, and `src/brain_image/configs/train_low_level.yaml` for the training config.

We also tried to use Dreamsim to train the low-level pipeline instead, but it didn't help. This part probably still works if the low-level pipeline is working, but currently it is not. See `src/brain_image/configs/train_low_level_dreamsim.yaml` for the training config.

**Second-Prior Alignment**: We tried outputting a second EEG aligning with our diffusion prior, and aligning the EEG latents after the prior, rather than before. The idea was to see if EEG alignment was interfering with training the diffusion prior, and if removing this interference would help. Eventually, we abandoned this track, and deleted the code. However, if this is something you want to look at in the future, you can find the code in [this commit](https://github.com/gasparyanartur/brain-image-implementation/blob/6d1cb137caf99b5459b87630a823bdbd6732fd25/src/brain_image/model/eeg_alignment.py).


## Setup

### Environments

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

**Note**: The training workflow is designed to work with WANDB for logging and experiment tracking. If you want to run without WANDB, set `enabled=false` in `src/brain_image/configs/wandb/wandb.yaml`. If you do use WANDB, make sure to set your API key in the `.env` file (see [Environment variables](#2-environment-variables)), and to log in with `wandb login` before running any scripts.

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

Build the base image first — this installs CUDA, Python 3.12, and PyTorch, and takes a while. 

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

*Note on AllJoined-16M:* Currently there is an issue in the code, causing the model to not learn anything when training on this dataset. We are investigating the issue, but in the meantime, we recommend using the Things-EEG2 dataset for training and evaluation. The AllJoined-16M dataset is still available in the codebase, but it might not work as expected until the issue is resolved. 
Also, we currently fetch the stim-order for this dataset in a hacky way, because it was not included in the original release. The dataset has since been updated to include the stim-order, but we have not yet updated our code to reflect that. 

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

### 7. Generate text captions (optional)

**Note on text captions:** This step is optional and only needed if you want to train with text alignment instead of image alignment. The text captions are generated from the stimulus images using a pretrained vision-language model.

There are two options for text captions: using a local VL model (which uses Qwen by default), or fetching from the HuggingFace API. Currently, only the local option works, because the HuggingFace API does not support processing this number of images in a reasonable time. 


```bash
# Local:
python scripts/data/generate_text_captions_local.py --config-name=generate_text_captions_local

# SLURM:
SBATCH_GROUP=gpu ssub generate_captions python scripts/data/generate_text_captions_local.py --config-name=generate_text_captions_local
```

Captions are written to `data/things-eeg2/captions/local.jsonl` (one JSON line per image with `path` and `caption` fields). Already-captioned images are skipped on re-runs. The model name and other settings are in `src/brain_image/configs/generate_text_captions_local.yaml`.

### 8. Generate text embeddings (optional)

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

