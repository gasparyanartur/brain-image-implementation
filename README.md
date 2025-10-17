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
```

* Option B: Go into src/brain_image/configs/ and update the paths in each configuration file to point to the directories you want to use.


### Step 2. Setup Environment

Next, setup the environment. There are two options for this: 

* Option A (Local): UV

Setting and install UV.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
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
 
The output should be an image named something like `images/brain_{datetime}.sif`. This is the image you will use for the slurm jobs.



### Step 3: Setup data

Once everything works, run the data downloading script. For this example we use sub-8, but repeat for any subs you want to include.

```
uv run scripts/download_data.py
```

Next, generate embeddings:

```
python scripts/generate_embeddings.py
```

### Step 4: Configs

Before you start training, you need to configure the hydra configs, found under `src/brain_image/configs/`.
In particular, make sure the paths are pointing to desired location, as mention in `Step 1`.



### Step 5: Logging

TODO: Talk about wandb, under configs/... Mention setting up WANDB_API_KEY in your environment 