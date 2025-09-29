#%%
import sys
sys.path.append('/mnt/vepfs/audio/lichang/HeartDiffuser/Heart_diffuser')
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
os.CUDA_VISIBLE_DEVICES = '0,1,2,3,4,5,6,7'
#%%
device = torch.device("cuda")
file_path = '/mnt/vepfs/audio/lichang/HeartDiffuser/Heart_diffuser/base_no_compress.yaml'
with open(file_path, 'r') as file:
    config = yaml.safe_load(file)


ecg = np.load("x_II_seg.npy")
ecg_noisy = AddNoise(ecg[:,None,:].copy(), SNR=15)
# ecg_missing, mask = imputation_pattern(ecg[:,None,:].copy(), extended = True)
ecg = torch.tensor(ecg[:,None,:].copy(), dtype=torch.float32)
ecg_missing = torch.tensor(ecg_noisy, dtype=torch.float32)

null = torch.zeros_like(ecg)

input_sig = torch.concat([null,null,ecg_missing,null], dim = 2)
Model = CSDI_base(config, device, L = 500*4)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load('no_compress799.pth'))

train_sig, test_sig = train_test_split(input_sig, test_size=0.2, random_state=42)

D_loader = DataLoader(test_sig, batch_size=1024, shuffle=False)
#%%
print('Generating ECG')
import time
for batch_index, batch in enumerate(D_loader):
    input_sig = batch.to(device)
    t1 = time.time()
    results = Model(input_sig, 100, '23', borrow_mode = 2, sample_steps=4, DDIM_flag=0)
    t2 = time.time()
    np.save(f"fifteenSNR{batch_index}_full.npy", results.detach().cpu().numpy())
    # np.save(f"IMP{batch_index}.npy", results.detach().cpu().numpy())
    print(t2 - t1)
#%%
