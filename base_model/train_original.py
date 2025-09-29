#%%
import torch
import torchaudio
import transformers
import scipy.io
import time
import os
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml
from diffusion_model_no_compress_final import diff_CSDI, CSDI_base
from utils_together_original import train
from self_process import imputation_pattern, AddNoise
os.CUDA_VISIBLE_DEVICES = '1,2,3,4,5,6,7' # Number of GPUs to use, you can change this based on your setup
device = torch.device("cuda")
signal = np.load('Final_sig_combined.npy')
print(signal.shape)
# Normalization the blood pressure signal, which cannot be simply rescaled to [-1,1] range
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
file_path = 'base_no_compress_original.yaml'
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
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, pin_memory=True, num_workers  = 8)

val_dataset = CustomSignalDataset(inputs_val, impute_val, noisy_val, mask_val)
val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = True)
#%%
from torch.nn.parallel import DistributedDataParallel as DDP
Model = CSDI_base(config, device, L = 500*4)
Model = torch.nn.DataParallel(Model).to(device)
train(Model,config['train'],train_loader,valid_loader=val_loader,valid_epoch_interval=10)
