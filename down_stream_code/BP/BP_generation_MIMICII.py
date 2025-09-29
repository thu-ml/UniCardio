#%% OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py
#https://github.com/pytorch/pytorch/issues/15808
import scipy.io
import time
import os
os.chdir('/root/autodl-tmp/BioDiffuser')
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
from diffusion_model_no_compress import diff_CSDI, CSDI_base
from utils_nin_zongheng import train
from self_process import imputation_pattern, AddNoise
import neurokit2 as nk
import pickle
device = torch.device("cuda")
#%%
signal = np.load('Final_sig_combined.npy')
N, channels, L = signal.shape
for i in range(signal.shape[0]):
    signal[i,1,:] = (signal[i,1,:]-100)/50
    
signal_impute = imputation_pattern(signal.copy(), extended = True)
signal_noisy = AddNoise(signal.copy(), SNR = 15)

signal = torch.tensor(signal, dtype = torch.float32)
mask = torch.tensor(signal_impute[1], dtype = torch.float32)
mask = mask[:,0,:]
signal_impute = torch.tensor(signal_impute[0], dtype = torch.float32)
signal_noisy = torch.tensor(signal_noisy, dtype = torch.float32)
file_path = 'base_no_compress.yaml'
with open(file_path, 'r') as file:
    config = yaml.safe_load(file)

signal = torch.reshape(signal, [signal.shape[0], 3*signal.shape[2]])
signal_impute = torch.reshape(signal_impute, [signal_impute.shape[0], 3*signal_impute.shape[2]])
signal_noisy = torch.reshape(signal_noisy, [signal_noisy.shape[0], 3*signal_noisy.shape[2]])

null = torch.zeros([signal.shape[0], int(signal.shape[1]/3)])

signal = torch.concatenate([signal, null], dim = -1)
signal_impute = torch.concatenate([signal_impute, null], dim = -1)
signal_noisy = torch.concatenate([signal_noisy, null], dim = -1)


test_size = 20000
validation_size = 20000

inputs_train, inputs_temp = train_test_split(
    signal, test_size=test_size + validation_size, random_state=42
)

# Now, split 'inputs_temp' into validation and test sets

inputs_val, inputs_test = train_test_split(
    inputs_temp, test_size=test_size, random_state=42
)



impute_train, impute_temp = train_test_split(
    signal_impute, test_size=test_size + validation_size, random_state=42
)

# Now, split 'inputs_temp' into validation and test sets

impute_val, impute_test = train_test_split(
    impute_temp, test_size=test_size, random_state=42
)


noisy_train, noisy_temp = train_test_split(
    signal_noisy, test_size=test_size + validation_size, random_state=42
)

# Now, split 'inputs_temp' into validation and test sets

noisy_val, noisy_test = train_test_split(
    noisy_temp, test_size=test_size, random_state=42
)

mask_train, mask_temp = train_test_split(
    mask, test_size=test_size + validation_size, random_state=42
)

# Now, split 'inputs_temp' into validation and test sets

mask_val, mask_test = train_test_split(
    mask_temp, test_size=test_size, random_state=42
)

#%%

from torch.utils.data import Dataset
class CustomSignalDataset(Dataset):
    def __init__(self, signal, signal_impute, signal_noisy, mask):
        self.signal = signal
        self.signal_impute = signal_impute
        self.signal_noisy = signal_noisy
        self.mask = mask

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, idx):
        return self.signal[idx, None, :], self.signal_impute[idx,None,:], self.signal_noisy[idx,None,:], self.mask[idx,None,:]
    
train_dataset = CustomSignalDataset(inputs_train, impute_train, noisy_train, mask_train)
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers  = 96, prefetch_factor=2)

inputs_gen = torch.concat([inputs_val, inputs_test], dim = 0)
impute_gen = torch.concat([impute_val, impute_test], dim = 0)
noisy_gen = torch.concat([noisy_val, noisy_test], dim = 0)
mask_gen = torch.concat([mask_val,mask_test], dim = 0)

gen_dataset = CustomSignalDataset(inputs_gen, impute_gen, noisy_gen, mask_gen)
gen_loader = DataLoader(gen_dataset, batch_size = 2048, shuffle = True)
#%%
Model = CSDI_base(config, device, L = 500*4)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load('no_compress799.pth'))

Model.eval()

# Directory to save the results
os.makedirs('MIMIC_II_results', exist_ok=True)

# Loop through the DataLoader and perform inference on each batch
for i, batch in enumerate(gen_loader):
    input_sig = batch[0].clone()
    start = time.time()
    results = Model(input_sig.to(device), 50, '02', borrow_mode=0)
    end = time.time()

    # Save the results for each batch
    results_path = f'results_batch_mimic{i}.pt'
    torch.save(results, results_path)
    
    results_path = f'input_batch_mimic{i}.pt'
    torch.save(input_sig, results_path)
    
    

    # Clear memory
    del input_sig, results
    torch.cuda.empty_cache()
    
    print(f'Batch {i} processed in {(end - start) / 60:.2f} minutes. Results saved to {results_path}')
