import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pylot as plt 
from torchsummary import summary 
import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

class SparseAutoencoder(nn.Module):
    def __init__(self, in_dims, h_dims, sparsity_lambda=1e-4, sparsity_target=0.05, xavier_norm_init=True):
        super().__init__()