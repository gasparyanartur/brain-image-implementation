<!-- Generated guidance for AI coding agents working on this repository -->
# Copilot instructions for brain-image-implementation

This file gives focused, actionable guidance for an AI coding agent to be productive in this repository.

- **Big picture:** Core package lives in `src/brain_image`. The code implements training and evaluation pipelines for mapping EEG to image latents. Major pieces:
  - `src/brain_image/model/` — model implementations and Lightning modules (see `model.py`, `eeg_alignment.py`, `comm_alignment.py`).
  - `src/brain_image/data/` — datamodules and dataset logic used by Lightning `TrainingModule`.
  - `src/brain_image/configs/` — Hydra configuration tree; training and dataset configs live here and are the authoritative runtime settings.
  - `scripts/` — convenience entrypoints and cluster helpers (embedding generation, setup, SLURM launch scripts, container build scripts in `scripts/container/`).

- **How training is launched:** Use the training script and Hydra config names. Example:

  ```bash
  python scripts/train_eeg.py --config-name=train_eeg dataset=things-eeg2
  ```

  - Configs are hierarchical. Override using Hydra CLI syntax, e.g. `eeg_encoder@model.eeg_encoder=atms` or `model.eeg_encoder.num_layers=6`.
  - Default config names are set with `@hydra.main` in each script — inspect the training script to see its default config group.

- **Environment & dependencies:** `pyproject.toml` lists major dependencies: PyTorch, Lightning, Hydra/OmegaConf, WandB, diffusers/transformers, dreamsim, etc. Python >= 3.12 is expected.

- **Data / storage conventions:**
  - Large artifacts are stored outside the repo: `data/`, `tensorcache/`, `logs/`, and `models/` (policy: symlink to cluster storage when on SLURM).
  - Precompute image latents into `tensorcache/` via `scripts/generate_embeddings.py` before training. See `src/brain_image/model/img_encoder.py` for supported encoders.

- **Logging & checkpoints:**
  - Lightning logging is configured in `src/brain_image/trainer.py`. Outputs go to `logs/train` by default; WandB is optional and controlled by `.env` (WANDB_API_KEY) and config group `wandb`.
  - Checkpoint naming, monitor metric, and early stopping are configurable via `TrainerConfig` in `trainer.py`.

- **Common code patterns and pitfalls for code edits:**
  - Lightning `TrainingModule` subclasses live under `src/brain_image/model/` and expect a `data_module` implementing `train/val/test` dataloaders.
  - Many modules use `torch.compile()` (see `model.py`) — avoid editing boundaries that would break compilation assumptions without re-testing.
  - Debugging prints are gated by env vars using the `is_debug_layer_active` pattern (env var `DEBUG_<LAYER>` or lowercase). Use that instead of ad-hoc prints.

- **Build, test, and run commands:**
  - Install dependencies via the project's packaging flow (see `pyproject.toml`) or use provided `uv`/Singularity scripts documented in `README.md`.
  - Data setup: `python scripts/setup_data.py --subs 8` (or use SLURM wrapper in `scripts/slurm/`).
  - Generate embeddings: `python scripts/generate_embeddings.py`.
  - Train: `python scripts/train_eeg.py --config-name=[CONFIG_NAME] dataset=[DATASET]`.
  - Tests: `pytest` (tests configured under `tests/` in `pyproject.toml`).

- **Config editing guidance:**
  - Always prefer CLI overrides for one-off experiments. For persistent changes, edit YAMLs in `src/brain_image/configs/` and keep path keys updated.
  - When adding new config groups, mirror existing patterns (group directories, schema classes in `configs.py`).

- **Integration points to be careful with:**
  - Hugging Face Hub (`huggingface_hub`), WandB, and any external model weights — ensure `.env` or cluster secrets are present before running.
  - Singularity container build scripts are in `scripts/container/` and may assume `sudo`/cluster resources.

- **Styling & formatting:**
  - `black` is configured with `line-length = 150` in `pyproject.toml`.
  - Tests live in `tests/` and follow pytest naming conventions from `pyproject.toml`.

- **Where to look for examples in this repo:**
  - `src/brain_image/trainer.py` — how logging, checkpointing, and WandB are configured.
  - `src/brain_image/model/model.py` — base `TrainingModule` and helper layers (debug, adapters, `torch.compile`).
  - `scripts/generate_embeddings.py`, `scripts/setup_data.py`, and `scripts/container/` — common operational scripts.

If anything important is missing from these instructions (for example, custom entrypoints, uncommon environment setup, or private registries), tell me what to inspect and I will update this document.
