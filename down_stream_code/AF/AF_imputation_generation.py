#%%
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
from ts2vg import NaturalVG
import yaml
from diffusion_model_no_compress import CSDI_base
from utils_together2 import train
from self_process import imputation_pattern, AddNoise
device = torch.device("cuda")
file_path = 'base_no_compress.yaml'
with open(file_path, 'r') as file:
    config = yaml.safe_load(file)

device = torch.device("cuda")
file_path = 'base_no_compress.yaml'
with open(file_path, 'r') as file:
    config = yaml.safe_load(file)

Model = CSDI_base(config, device, L = 500*4)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load('no_compress799.pth'))
#%%

all_results = []
all_labels = []
all_truth = []
ppg = []
Model.eval()
# Load and process data
with torch.no_grad():
    for i in range(0, 18):
        truth_name = f'input_batch_large{i}.pt'
        truth = torch.load(truth_name)[:,0,1000:1500][:,None,:].detach().cpu()
        noisy_truth,mask = imputation_pattern(truth, extended = True)
        null = torch.zeros_like(truth)
        input_sig = torch.concat([null,null,noisy_truth,null], dim = 2).to(device)
        results = Model(input_sig.to(device), 100, '23', borrow_mode=2)
        torch.save(results.detach().cpu(), f"imputed{i}.pt")
        torch.save(mask, f"mask{i}.pt")

    
