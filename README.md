# Brain Image Implementation

(TODO: Update this)

## Project Description

(TODO: Update this)



## How to run this

The first step is to make sure the necessary folders exist in the repository. There are two abouts about this:

* Option A (Recommended): Setup the following folders in your repository:

```
data/
tensorcache/
logs/
model/
```

If you are working on a cluster, make sure to symlink these to your storage volume, as they will contain large volumes of data.

```
storage_path=$PATH_TO_VOLUME_STORAGE
symlink $storage_path/data data
symlink $storage_path/tensorcache tensorcache
symlink $storage_path/logs logs
symlink $storage_path/models models
```

* Option B: Go into src/brain_image/configs/ and update the paths in each configuration file to point to the directories you want to use.


Next, install and setup uv:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv sync
```

Try importing torch in a Python shell to confirm installation.

Once everything works, run the data downloading script. For this example we use sub-8, but repeat for any subs you want to include.

```
uv run scripts/download_data.py
```

Next, generate embeddings:

```
uv run scripts/gen_embeddings.py
```

Before you start training, you need to configure the hydra configs, found under `src/brain_image/configs/`.
In particular, make sure the paths are pointing 

```
uv run scripts/train_nice.py nice_config.model_name=synclr
uv run scripts/train_nice.py nice_config.model_name=aligned_synclr
```

Under `logs/` there should be a `synclr` and `aligned_synclr` directory. In each, find the checkpoint and copy the relative path to it. For instance: `logs/aligned_synclr/version_0/checkpoints/checkpoint/epoch=00-val/loss=5.95.ckpt`.

Now, evaluate it:

```
uv run scripts/evaluate_nice.py checkpoint_path=PATH_TO_SYNCLR_CHECKPOINT
uv run scripts/evaluate_nice.py checkpoint_path=PATH_TO_ALIGNED_SYNCLR_CHECKPOINT
```

The evaluated results should be printed in the console.

## Weights & Biases Integration

This project includes integration with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization. To use wandb:

### Setup

1. **Install and authenticate wandb:**
   ```bash
   python scripts/setup_wandb.py
   ```
   This script will:
   - Install wandb if not already installed
   - Guide you through the login process
   - Create a configuration file

2. **Configure your wandb settings:**
   Edit the generated `wandb_config.yaml` file with your preferences:
   ```yaml
   wandb_entity: your_username_or_team
   wandb_project: brain-image-nice
   wandb_log_model: false
   wandb_tags: []
   ```

### Usage

1. **Enable wandb in your training configuration:**
   ```yaml
   # In your trainer config (e.g., src/brain_image/configs/trainer/nice_trainer.yaml)
   enable_wandb: true
   wandb_project: brain-image-nice
   wandb_entity: your_username_or_team
   wandb_log_model: false
   wandb_tags: ["experiment", "nice"]
   ```

2. **Run training with wandb logging:**
   ```bash
   uv run scripts/train_nice.py nice_config.model_name=aligned_synclr
   ```

3. **Test wandb integration:**
   ```bash
   python scripts/test_wandb.py
   ```

### What gets logged

With wandb enabled, the following information will be automatically logged:
- Training and validation metrics (loss, accuracy, etc.)
- Model hyperparameters
- Training configuration
- System information (GPU usage, memory, etc.)
- Model checkpoints (if `wandb_log_model: true`)

### Viewing results

After training, you can view your experiments in the wandb dashboard at [wandb.ai](https://wandb.ai). Navigate to your project to see:
- Training curves and metrics
- Model comparison tables
- System resource usage
- Experiment configurations