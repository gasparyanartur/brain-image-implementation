import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image


from pathlib import Path 
if Path.cwd().name != "Generation":
    os.chdir("Generation")


train = False
classes = None
pictures= None

data_dir = "../data/things-eeg2"
images_dir = os.path.join(data_dir, "images_set")

def load_data():
    data_list = []
    label_list = []
    texts = []
    images = []
    
    text_directory = os.path.join(images_dir, "training_images" if train else "test_images")

    dirnames = [d for d in os.listdir(text_directory) if os.path.isdir(os.path.join(text_directory, d))]
    dirnames.sort()
    
    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    for dir in dirnames:

        try:
            idx = dir.index('_')
            description = dir[idx+1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue
            
        new_description = f"{description}"
        texts.append(new_description)

    img_directory = os.path.join(images_dir, "training_images" if train else "test_images")
    
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()

    if classes is not None and pictures is not None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else:

        print("Error")
    return texts, images
texts, images = load_data()
# images





import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
import itertools
import math
import re
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
import numpy as np
from loss import ClipLoss
import argparse
from torch import nn
from torch.optim import AdamW


class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250                 # Sequence length
        self.pred_len = 250                # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Dimension of the feedforward network
        self.activation = 'gelu'           # Activation function
        self.enc_in = 63                   # Encoder input dimension (example value)

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False,  num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]      
        # print("enc_out", enc_out.shape)
        return enc_out

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class ATMS(nn.Module):    
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
         
    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        # print(f'After attention shape: {x.shape}')
        # print("x", x.shape)
        # x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out  


def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None
    

def get_eegfeatures(sub, eegmodel, dataloader, device, text_features_all, img_features_all, k, train, models_dir):
    eegmodel.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha =0.9
    top5_correct = 0
    top5_correct_count = 0

    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    ridge_lambda = 0.1
    save_features = True
    features_list = []  # List to store features    
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_id = extract_id_from_string(sub)
            # eeg_data = eeg_data.permute(0, 2, 1)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            # if not config.insubject:
            #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)          
            eeg_features = eeg_model(eeg_data, subject_ids)
            features_list.append(eeg_features.detach().cpu())

        
            logit_scale = eeg_model.logit_scale 
                   
            regress_loss =  mse_loss_fn(eeg_features, img_features)
            # print("eeg_features", eeg_features.shape)
            # print(torch.std(eeg_features, dim=-1))
            # print(torch.std(img_features, dim=-1))
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = (regress_loss + ridge_lambda * l2_norm)       
            img_loss = eegmodel.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eegmodel.loss_func(eeg_features, text_features, logit_scale)
            contrastive_loss = img_loss
            # loss = img_loss + text_loss

            regress_loss =  mse_loss_fn(eeg_features, img_features)
            # print("text_loss", text_loss)
            # print("img_loss", img_loss)
            # print("regress_loss", regress_loss)            
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = (regress_loss + ridge_lambda * l2_norm)       
            loss = alpha * regress_loss *10 + (1 - alpha) * contrastive_loss*10
            # print("loss", loss)
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):

                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                

                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                # logits_single = (logits_text + logits_img) / 2.0
                logits_single = logits_img
                # print("logits_single", logits_single.shape)

                # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                if predicted_label == label.item():
                    correct += 1        
                total += 1

        features_tensor = torch.cat(features_list, dim=0)
        if save_features:
            print("features_tensor", features_tensor.shape)
            torch.save(features_tensor.cpu(), os.path.join(models_dir, f"ATM_S_eeg_features_{sub}-train.pt" if train else f"ATM_S_eeg_features_{sub}-test.pt"))  # Save features as .pt file
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy, labels, features_tensor.cpu()

from IPython.display import Image, display
config = {
    "data_path": os.path.join(data_dir, "Preprocessed_data_250Hz"),
    "project": "atms_reconstruction",
    "entity": "gasparyanartur",
    "name": "lr=3e-4_img_pos_pro_eeg",
    "lr": 3e-4,
    "epochs": 50,
    "batch_size": 1024,
    "logger": True,
    "encoder_type":'ATMS',
}

model_name = 'generated_mine_external' 
models_dir = os.path.join("models", model_name)
os.makedirs(models_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = config['data_path']
features_dir = os.path.join(data_dir, "features")
emb_img_test = torch.load(os.path.join(features_dir, 'ViT-H-14_features_test.pt'))
emb_img_train = torch.load(os.path.join(features_dir, 'ViT-H-14_features_train.pt'))

eeg_model = ATMS()
print('number of parameters:', sum([p.numel() for p in eeg_model.parameters()]))

#####################################################################################

def load_external_eeg_cp(cp_path):
    cp = torch.load(cp_path)
    state_dict = cp["state_dict"]
    filtered_state_dict = {k[len("eeg_encoder."):]:v for k, v in state_dict.items() if k.startswith("eeg_encoder.")}
    print(f"Loaded {len(filtered_state_dict)} layers")
    return filtered_state_dict

#cp = torch.load("models/generated_original/ATM_S_eeg_features_sub-08-train.pt")
#cp = torch.load("models/contrast/ATMS/sub-08/09-29_10-36/40.pth")
#cp = load_external_eeg_cp("tmp/train_eeg-epoch=20-VAL__loss=1.8497.ckpt")
#cp = load_external_eeg_cp("tmp/train_eeg-epoch=39-VAL__loss=6.7002.ckpt")
cp = load_external_eeg_cp("tmp/train_eeg-epoch=32-VAL__loss=2.0830.ckpt")

eeg_model.load_state_dict(cp)
eeg_model = eeg_model.to(device)
sub = 'sub-08'
#####################################################################################



import multiprocessing as mp
#num_workers = mp.cpu_count()
num_workers = 0

#####################################################################################
train_dataset = EEGDataset(data_path, subjects= [sub], train=True)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)
text_features_test_all = train_dataset.text_features
img_features_test_all = train_dataset.img_features

train_loss, train_accuracy, labels, eeg_features_train = get_eegfeatures(sub, eeg_model, train_loader, device, text_features_test_all, img_features_test_all, models_dir=models_dir, k=200, train=True)
print(f" - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")


test_dataset = EEGDataset(data_path, subjects= [sub], train=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)
text_features_test_all = test_dataset.text_features
img_features_test_all = test_dataset.img_features
test_loss, test_accuracy,labels, eeg_features_test = get_eegfeatures(sub, eeg_model, test_loader, device, text_features_test_all, img_features_test_all, models_dir=models_dir, k=200, train=False)
print(f" - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
#####################################################################################


import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import open_clip
from matplotlib.font_manager import FontProperties

import sys
from diffusion_prior import *
from custom_pipeline import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


emb_img_train_4 = emb_img_train["img_features"].view(1654,10,1,1024).repeat(1,1,4,1).view(-1,1024)
emb_eeg = torch.load(os.path.join(models_dir, 'ATM_S_eeg_features_sub-08-train.pt'))
emb_eeg_test = torch.load(os.path.join(models_dir, 'ATM_S_eeg_features_sub-08-test.pt'))

TRAIN_DIFFUSION_PRIOR = True
if TRAIN_DIFFUSION_PRIOR:
    assert "original" not in models_dir

save_path = os.path.join(models_dir, f'{config["encoder_type"]}/{sub}/diffusion_prior.pt')

dataset = EmbeddingDataset(
    c_embeddings=eeg_features_train, h_embeddings=emb_img_train_4, 
    # h_embeds_uncond=h_embeds_imgnet
)
dl = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=64)
diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
# number of parameters
print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
pipe = Pipe(diffusion_prior, device=device)

if TRAIN_DIFFUSION_PRIOR:
    pipe.train(dl, num_epochs=150, learning_rate=1e-3) # to 0.142
    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)
    torch.save(pipe.diffusion_prior.state_dict(), save_path)

else:
    pipe.diffusion_prior.load_state_dict(torch.load(save_path, map_location=device))

pipe.diffusion_prior.eval()
pipe.diffusion_prior.requires_grad_(False)

# Generate test embeddings

test_embeddings = []
with torch.no_grad():
    for k in tqdm(list(range(200)), desc="Generating test embeddings..."):
        eeg_embeds = emb_eeg_test[k:k+1]
        h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=5.0)
        test_embeddings.append(h.detach().cpu())
    test_embeddings = torch.cat(test_embeddings)
print(test_embeddings.shape)

print("All tensors and modules:")
import itertools as it
to_delete = []

items = locals().items()
try:
    for k, v in items:
        if isinstance(v, (torch.Tensor, torch.nn.Module)):
            v.to("cpu")
            print(k.replace('\n', ''))
        if k.startswith("_"):
            to_delete.append((k, v))
            
    for (k, v) in to_delete:
        try:
            del v
        except NameError:
            pass
except RuntimeError as e:
    pass

import gc
try:
    del eeg_model
except NameError:
    pass
try:
    del generator
except NameError:
    pass

try:
    del pipe
except NameError:
    pass

try:
    del diffusion_prior
except NameError:
    pass

gc.collect()
torch.cuda.empty_cache()

from pathlib import Path
import PIL
import torchvision.transforms.v2 as tv2


from PIL import Image
import os
import diffusers 

# Not sure why they've added this, but it makes the model fail
if "http_proxy" in os.environ:
    os.environ.pop("http_proxy")
if "https_proxy" in os.environ:
    os.environ.pop("https_proxy")
os.environ["HF_HUB_OFFLINE"] = "False"


# Assuming generator.generate returns a PIL Image
generator = Generator4Embeds(num_inference_steps=4, device=device)

diffusers.utils.logging.disable_progress_bar()

directory = f"generated_imgs/{model_name}/{sub}"
os.makedirs(directory, exist_ok=True)
NUM_GEN_PER_IMG = 3

selected_figures = [
    "seaweed", "pug", "scallop", "slide", "dreidel", "pajamas", "jelly_bean", "possum", "wine"
]

idx_to_fig = {}
for i, text in enumerate(texts):
    for fig_text in selected_figures:
        if text == fig_text:
            idx_to_fig[i] = fig_text


selected_gens = []
selected_gts = []
selected_texts = []

test_imgs_path = Path(f"../data/things-eeg2/images_set/test_images")
for idx, text in idx_to_fig.items():
    gt_paths = list(test_imgs_path.rglob(f"{text}_*.jpg"))
    gt_path = gt_paths[0]
    gt = PIL.Image.open(gt_path)
    all_gens = [tv2.functional.pil_to_tensor(gt.resize((512, 512), PIL.Image.Resampling.BICUBIC))]
    for g in range(NUM_GEN_PER_IMG):
        h = test_embeddings[idx:idx+1].to(dtype=torch.float16)
        out = tv2.functional.pil_to_tensor(generator.generate(h))
        all_gens.append(out)
    gens = torch.stack(all_gens)

    selected_gens.append(gens)
    selected_texts.append(text)
selected_gens = torch.stack(selected_gens)

selected_gens.shape

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import PIL

fig = plt.figure(figsize=(8., 19.))
fig.suptitle(f"Selected Results for model {model_name}")
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(9, 4),  # creates 2x2 grid of Axes
                 axes_pad=0,  # pad between Axes in inch.
                
                )

titles = ["Ground Truth", *(f"Generation {i}" for i in range(1, 4)) ]
for i_row in range(9):
    for i_col in range(4):
        ax = grid[4*i_row + i_col] 
        ax.imshow(selected_gens[i_row, i_col].permute(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        if i_row == 0:
            ax.set_title(titles[i_col])

fig.tight_layout()
fig.savefig(os.path.join(directory, "plot.png"))
plt.show()