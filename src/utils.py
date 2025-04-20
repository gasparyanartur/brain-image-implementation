import torch
import os


if "DEVICE" in os.environ:
    DEVICE = torch.device(os.environ["DEVICE"])
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "DTYPE" in os.environ:
    DTYPE = torch.dtype(os.environ["DTYPE"])
else:
    DTYPE = torch.float16
