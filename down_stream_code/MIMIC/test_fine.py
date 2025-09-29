# %%
import torch
import scipy.io
import time
import os
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml
from diffusion_model_no_compress_finetune import diff_CSDI, CSDI_base
from utils_finetune import train
from self_process import imputation_pattern, AddNoise
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
os.CUDA_VISIBLE_DEVICES = '1,2,3,4,5,6,7' # Number of GPUs to use, you can change this based on your setup
device = torch.device("cuda")
signal_dummy = torch.load('mimic_cleaned.pth')
signal = torch.zeros([signal_dummy.shape[0], 3, 500])
signal[:,0,:] = signal_dummy[:,0:500]
signal[:,1,:] = signal_dummy[:,500:1000]
signal[:,2,:] = signal_dummy[:,1000:1500]
signal = signal.numpy()
subject_ids = np.load('mimic_cleaned_label.npy')
print(signal.shape)






    
# N, channels, L = signal.shape
# for i in range(signal.shape[0]):
#     signal[i,1,:] = signal[i,1,:]*50 +100

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


# Leave-subject-out split
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(gss_test.split(signal, groups=subject_ids))

# Create temporary datasets for splitting
temp_signal = signal[train_val_idx]
temp_subject_ids = subject_ids[train_val_idx]

# Split train_val into train and validation
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(gss_val.split(temp_signal, groups=temp_subject_ids))

# Get original indices
original_train_idx = train_val_idx[train_idx]
original_val_idx = train_val_idx[val_idx]

# Partition the data
inputs_train = signal[original_train_idx]
inputs_val = signal[original_val_idx]
inputs_test = signal[test_idx]

impute_train = signal_impute[original_train_idx]
impute_val = signal_impute[original_val_idx]
impute_test = signal_impute[test_idx]

noisy_train = signal_noisy[original_train_idx]
noisy_val = signal_noisy[original_val_idx]
noisy_test = signal_noisy[test_idx]

mask_train = mask[original_train_idx]
mask_val = mask[original_val_idx]
mask_test = mask[test_idx]

#%%


from torch.utils.data import Dataset
class CustomSignalDataset(Dataset):
    def __init__(self, signal, signal_impute, signal_noisy, mask, subject):
        self.signal = signal
        self.signal_impute = signal_impute
        self.signal_noisy = signal_noisy
        self.mask = mask
        self.subject = subject

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, idx):
        return self.signal[idx, None, :], self.signal_impute[idx,None,:], self.signal_noisy[idx,None,:], self.mask[idx,None,:], self.subject[idx]



train_dataset = CustomSignalDataset(inputs_train, impute_train, noisy_train, mask_train, subject_ids[train_idx])
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, pin_memory=True, num_workers  = 8)

val_dataset = CustomSignalDataset(inputs_val, impute_val, noisy_val, mask_val, subject_ids[val_idx])
val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = True)

test_dataset = CustomSignalDataset(inputs_test, impute_test, noisy_test, mask_test, subject_ids[test_idx])
test_loader = DataLoader(test_dataset, batch_size = 2048, shuffle = True)



#%%
# --- Model Loading ---


with open('base_no_compress_original.yaml', 'r') as file: config = yaml.safe_load(file)
Model = CSDI_base(config, device, L=500 * 4).to(device)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load("check/model_fine_0.pth"), strict=False)#which checkpoint you want

batch = next(iter(test_loader))
print("Running model inference (this may take a while)...")
Model.eval()
batch0 = batch[0][:, 0, None, :]
batch_subjects = batch[4]
with torch.no_grad():
    results = Model(batch0.to(device), n_samples=50 , model_flag='21', borrow_mode=2, DDIM_flag=0, sample_steps=6, train_gen_flag=1)
#%%
mean_results = (results.mean(dim=1).detach().cpu().numpy()[:, 0, :])*50 +100

truth = (batch[0][:, 0, 500:1000].detach().cpu().numpy())


rmse = []
for index in range(truth.shape[0]):
    e = np.sqrt(((mean_results[index,:] - truth[index])**2).sum()/500)
    rmse.append(e)
rmse = np.array(rmse)
print("RMSE Mean: ", rmse.mean())
print("RMSE Std: ", rmse.std()) 


mae = []
for index in range(truth.shape[0]):
    e = np.absolute((mean_results[index] - truth[index])).sum()/500
    mae.append(e)

mae = np.array(mae)
print("MAE Mean: ", mae.mean())
print("MAE Std: ", mae.std())

import scipy.stats as stats
def analyze_ecg_distribution_difference(signal1, signal2):
    ks_stat, ks_pvalue = stats.ks_2samp(signal1, signal2)  
    return ks_stat, ks_pvalue

KS = []
for index in range(truth.shape[0]):
    m = batch[3][index,0,:].detach().numpy()
    s1 = mean_results[index, :]
    s2 = truth[index, :]
    ks_stat, ks_pvalue = analyze_ecg_distribution_difference(s1, s2)
    KS.append(ks_stat)
KS = np.array(KS)
print("KS Mean: ", KS.mean())
print("KS Std: ", KS.std())

# Calculate Systolic and Diastolic Blood Pressure
print("\n--- Blood Pressure Analysis ---")
#%%
index = 2
plt.plot(mean_results[index, :], label='Predicted Signal')
plt.plot(truth[index, :], label='Ground Truth Signal')
#%%
print(10)
# Calculate systolic (max) and diastolic (min) BP for predicted signals
pred_systolic = np.max(mean_results, axis=1)
pred_diastolic = np.min(mean_results, axis=1)

# Calculate systolic (max) and diastolic (min) BP for ground truth signals
truth_systolic = np.max(truth, axis=1)
truth_diastolic = np.min(truth, axis=1)

# Calculate errors for systolic BP
systolic_errors = np.abs(pred_systolic - truth_systolic)
systolic_rmse = np.sqrt(np.mean((pred_systolic - truth_systolic)**2))
systolic_mae = np.mean(systolic_errors)

# Calculate errors for diastolic BP

diastolic_errors = np.abs(pred_diastolic - truth_diastolic)
diastolic_rmse = np.sqrt(np.mean((pred_diastolic - truth_diastolic)**2))
diastolic_mae = np.mean(diastolic_errors)

print(f"Systolic BP - RMSE: {systolic_rmse:.2f} mmHg, MAE: {systolic_mae:.2f} mmHg")
print(f"Diastolic BP - RMSE: {diastolic_rmse:.2f} mmHg, MAE: {diastolic_mae:.2f} mmHg")

# Display statistics
print(f"\nPredicted Systolic BP - Mean: {pred_systolic.mean():.2f} ± {pred_systolic.std():.2f} mmHg")
print(f"Ground Truth Systolic BP - Mean: {truth_systolic.mean():.2f} ± {truth_systolic.std():.2f} mmHg")
print(f"Predicted Diastolic BP - Mean: {pred_diastolic.mean():.2f} ± {pred_diastolic.std():.2f} mmHg")
print(f"Ground Truth Diastolic BP - Mean: {truth_diastolic.mean():.2f} ± {truth_diastolic.std():.2f} mmHg")

# Calculate correlation coefficients
systolic_corr = np.corrcoef(pred_systolic, truth_systolic)[0, 1]
diastolic_corr = np.corrcoef(pred_diastolic, truth_diastolic)[0, 1]
print(f"\nCorrelation - Systolic: {systolic_corr:.3f}, Diastolic: {diastolic_corr:.3f}")

