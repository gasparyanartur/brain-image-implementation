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
To keep the scope manageable, I simply use SynCLR, since this offers the best reported performance in their evaluations.
In my implementation, I import the aligned (pretrained) Dreamsim model along with the base SynCLR model 

### EEG Encoder


### Results

I trained both models with 256

|metric|unaligned_synclr|aligned_synclr|
|---|---|---|
|nice_loss|1.861|1.164
|top1_acc|0.477|0.650
|top3_acc|0.705|0.868
|top5_acc|0.799|0.923


## Reflections

- 

## How to run this

