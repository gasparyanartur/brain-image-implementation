# Brain Image Implementation

In this project, I reimplement the code for the paper [Human-Aligned Image Models Improve Visual Decoding from the Brain](https://arxiv.org/abs/2502.03081).

## Project Description

The paper tackles the problem of decoding images from brain activity. 
Recent approaches do this by aligning embeddings from EEG signals to embeddings of images.
To do this, three components are required:

1. A brain-signal encoder, which converts EEG signals to high-dimensional embeddings
2. A pretrained-image encoder, mapping images to imbeddings
3. A self-supervised loss function which encourages a mapping similar brain-image pairs close together, and different ones far apart. 

This paper proposes a method to better decode EEG signals into visual images by using human-aligned image encoders rather than more generic versions. They find that doing so increases the cosine similarity of similar samples, and decreases it for differing ones, which verifies that the mapping is indeed better.

### Data

For data, they use the [Things2 EEG dataset](https://www.sciencedirect.com/science/article/pii/S1053811922008758) for training and evaluation. 
In my recreation, I use the preprocessed data available on [osf](https://osf.io/3jk45/), along with the image pairs. 
They also run evaluations on [another data source](https://elifesciences.org/articles/82580), but I refrain from doing so to keep the scope limited.
The code for data loading and processing can be found in `src/data.py`.

### Image Encoder

They use [Dreamsim](https://arxiv.org/pdf/2306.09344), comparing the performance of various backbones models to the corresponding aligned versions. Specifically, they report performance for CLIP, OpenCLIP, DINO, DINOv2, and SynCLR.
To keep the scope manageable, I simply use [SynCLR](https://arxiv.org/abs/2312.17742), since this offers the best reported performance in their evaluations.
In my implementation, I import the aligned (pretrained) Dreamsim model along with the base SynCLR model from the [Dreamsim package](https://github.com/ssundaram21/dreamsim).


### EEG Encoder

In their paper, they use [NICE-EEG](https://arxiv.org/abs/2308.13234) to encode EEG signals into embeddings. Similarly, I implement the EEG-Encoder from the [authors' repository](https://github.com/eeyhsong/NICE-EEG), as found in `src/model.py`. Since the original authors use a data set with high dimensionality (250Hz), they employ aggressive pooling to reduce dimensionality. In my case, the data has lower dimensionaltity, and so I refrain from pooling, instead adding an additional convolution layer to get a similar output dimension.

### Training

During training, the image embeddings are precomputed, since the image encoders are frozen during training. 
I do the same, as seen in `src/model.py`. 

The models are trained using the [InfoNCE loss](https://arxiv.org/abs/1807.03748).
Specifically, a similarity matrix is formed between the dimensions of the image- and EEG embeddings.
Then, a categorical cross-entropy loss is applied between rows/columns and the corresponding index.
The idea is that the similarity should only be high on the same dimension for both embeddings, and low otherwise.
This trains the projection layers to map EEG embeddings and image embeddings to the same space.


### Results

I use 768 for both EEG- and image latents. They then get projected to a 256-dimensional space.

I train for 100 epochs with a learning rate of 8e-3 with a cosine annealing scheduler going down to 1e-4. 
To give the projector layers some time to stabilize, I put a longer lr warmup on the encoder layers: 2 on the projector layers and 4 on the encoder layers.

After training, I evaluate the top1, top3, and top5 accuracy on the test set, as the paper does.

|metric|unaligned_synclr|aligned_synclr|
|---|---|---|
|nice_loss|1.861|1.164
|top1_acc|0.477|0.650
|top3_acc|0.705|0.868
|top5_acc|0.799|0.923

The results, 65.0% top1-accuracy on Dreamsim and 47.7% on Base-SynCLR confirms the paper's findings that human aligned vision models are indeed better.
Intristingly, my accuracy seems to be marginally higher than that reported in the paper. 
It could be that I train on the entire training set, rather than partioning into train/validation sets. 
Alternativly, perhaps my particular choice of parameters happen to work better on that dataset.


## How to run this

Download the preprocessed eeg and imgs-latents from [Things-EEG2](https://osf.io/3jk45/). By default, the folder structure is expected to be:

```
data/
    things-eeg2/
        eeg/
            sub-01/
                ...
            ...
        imgs/
            training_images/
                ... 
            ...   
```

Next, install and setup uv:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv sync
```

Try importing torch in a Python shell to confirm installation.

Once everything works, run the model downloading script:

```
uv run scripts/download_models.py
```

Next, generate embeddings:

```
uv run scripts/gen_embeddings.py
```

Next it's time to train. First the unaligned, then the aligned model:

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