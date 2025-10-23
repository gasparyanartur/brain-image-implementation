# Brain Image Implementation

(TODO: Update this)

## Project Description

(TODO: Update this)



## How to run this

### Step 1. Setup directory

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
symlink $storage_path/.cache .cache
symlink $storage_path/experiments experiments
```

If you're on a SLURM cluster, also create the following directories:

```
mkdir -p logs/slurm/setup_data
mkdir -p logs/slurm/generate_embeddings
mkdir -p logs/slurm/train_eeg
mkdir -p logs/slurm/test_eeg
mkdir -p logs/slurm/sweep
```


* Option B: Go into src/brain_image/configs/ and update the paths in each configuration file to point to the directories you want to use.


### Step 2. Setup Environment

Next, setup the environment. There are two options for this: 

* Option A (Local): UV

Setting and install UV.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync
```

Try importing torch in a Python shell to confirm installation.

* Option B (SLURM): Singularity

I've setup utility scripts to make installation and configuration of images easier. They can be found under `scripts/container/`. 
The build script `scripts/build_singularity.sh` is a utility script that automatically chooses reasonable defaults. Your images will be build under `images/` unless specified otherwise.

First, you'll need to setup the base image (takes a while):

```
DEFINITION_FILE=scripts/container/singularity_base.def IMAGE_FILE=images/singularity_base.sif  ./scripts/container/build_singularity.sh
```

Once that is done, you should see a new image under `images/singularity_base.sif`. Next, you'll setup the main image.

```
./scripts/container/build_singularity.sh
```
 
The output should be an image named something like `images/brain_{datetime}.sif`. This is the image you will use for the slurm jobs. (Note, you might want to build locally and rsync to the cluster since you need sudo access.)


### Step 3: Setup data

Once everything works, run the data downloading script. For this example we use sub-8, but repeat for any subs you want to include.

```
python scripts/setup_data.py --subs 8
```
or
```
sbatch scripts/slurm/run_setup_data.sh --subs 8
```

### Step 4: Generate embeddings

The training loop assumes all relevant image latents are precomputed and stored in `tensorcache`. All list of supported image encoder latents can be found in `src/brain_image/model/img_encoder.py`.

By default, the embeddings generated are found in `src/configs/generate_embeddings.yaml`

```
python scripts/generate_embeddings.py 
```
or
```
sbatch scripts/slurm/run_generate_embeddings.sh
```

Embeddings for specific image encoders can be generated my specifying the `model_names` argument:

```
python scripts/generate_embeddings.py model_names=[clip_vith14]
```

### Step 4: Configs

Before you start training, you need to configure the hydra configs, found under `src/brain_image/configs/`.
In particular, make sure the paths are pointing to desired location, as mention in `Step 1`.

### Step 5: Environment

The training script loads your API keys from .env by default.
This is done mainly for two things: `Hugging Face Hub` and `Weights and Biases`

Create a .env file in the root of the project,.

TODO: Talk about wandb, under configs/... Mention setting up WANDB_API_KEY in your environment 