#%%
import sys
import scipy
import matplotlib.pyplot as plt
import numpy as np
import torch
import neurokit2 as nk
import scipy.io
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
import yaml
from diffusion_model_no_compress import CSDI_base
from utils_together2 import train
from self_process import imputation_pattern, AddNoise
device = torch.device("cuda")
file_path = 'base_no_compress.yaml'
with open(file_path, 'r') as file:
    config = yaml.safe_load(file)

Model = CSDI_base(config, device, L = 500*4)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load('no_compress799.pth'))

for s_id in range(2,18):
    if s_id == 8:
        continue
    if s_id == 12:
        continue
    ppg = scipy.io.loadmat(f'Subject{s_id}_PPG.mat')['PPG'][0,:]
    ecg = scipy.io.loadmat(f'Subject{s_id}_ECG.mat')['ECG']

    
    ecg = nk.ecg_clean(ecg[:,0], sampling_rate=125, method="neurokit",powerline = 50)

    length = int(np.floor(ppg.shape[0]/500))
    ppg = ppg[0:500*length].reshape((length, 500))
    ecg = ecg[0:500*length].reshape((length, 500))

    for i in range(length):
        ppg[i] = (ppg[i] - ppg[i].mean())/(ppg[i].std())
        ppg[i] = -1+2*(ppg[i] - ppg[i].min())/(ppg[i].max() - ppg[i].min())

        ecg[i] = (ecg[i] - ecg[i].mean())/(ecg[i].std())
        ecg[i] = -1+2*(ecg[i] - ecg[i].min())/(ecg[i].max() - ecg[i].min())
    ppg = torch.tensor(ppg.copy(), dtype = torch.float32)    
    ecg_corrupted = imputation_pattern(ecg[:,None,:].copy(), extended = True)
    ecg_noisy = AddNoise(ecg[:,None,:].copy(), SNR=10)

    ecg = torch.tensor(ecg[:,None,:].copy(), dtype=torch.float32)
    ecg_corrupted = torch.tensor(ecg_corrupted[0], dtype=torch.float32)
    ecg_noisy = torch.tensor(ecg_noisy, dtype=torch.float32)
    null = torch.zeros_like(ecg)

    input_sig = torch.concat([ppg[:,None,:],null,ecg_corrupted,null], dim = 2)

    start = time.time()
    Model.eval()
    results = Model(input_sig.to(device), 100, '23', borrow_mode = 2, sample_steps=4, DDIM_flag=0)
    end = time.time()
    print((end-start)/60)
    r = results.detach().cpu().numpy()
    np.save(f'results{s_id}_imp.npy', r)
    
    input_sig = torch.concat([ppg[:,None,:],null,ecg_noisy,null], dim = 2)
    start = time.time()
    Model.eval()
    results = Model(input_sig.to(device), 100, '23', borrow_mode = 2, sample_steps=4, DDIM_flag=0)
    end = time.time()
    print((end-start)/60)
    r = results.detach().cpu().numpy()
    np.save(f'results{s_id}_de.npy', r)

    print(s_id)
