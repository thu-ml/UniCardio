#%%
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

import numpy as np
import torch
from scipy.signal import find_peaks, find_peaks_cwt, peak_widths
from scipy.ndimage import binary_dilation, binary_erosion
import scipy.signal as signal_s
from sklearn.preprocessing import StandardScaler
import pickle
import os

class SimpleSignalMaskGenerator:
    def __init__(self, 
                 percentile_threshold=75,  # Use percentiles instead of prominence
                 expansion_width=10,
                 important_weight=3.0,
                 base_weight=1.0):
        """
        Simple mask generator - NO NORMALIZATION!
        Uses percentile-based thresholds instead of prominence
        """
        self.percentile_threshold = percentile_threshold
        self.expansion_width = expansion_width
        self.important_weight = important_weight
        self.base_weight = base_weight
    
    def detect_peaks_valleys_simple(self, signal_1d):
        """Simple peak/valley detection using percentiles"""
        # Calculate thresholds based on signal's own distribution
        high_threshold = np.percentile(signal_1d, self.percentile_threshold)
        low_threshold = np.percentile(signal_1d, 100 - self.percentile_threshold)
        
        # Find peaks (high values)
        peak_candidates = np.where(signal_1d > high_threshold)[0]
        
        # Find valleys (low values)  
        valley_candidates = np.where(signal_1d < low_threshold)[0]
        
        # Remove consecutive points - keep only local maxima/minima
        peaks = []
        if len(peak_candidates) > 0:
            peaks.append(peak_candidates[0])
            for i in range(1, len(peak_candidates)):
                if peak_candidates[i] - peak_candidates[i-1] > 5:  # Minimum separation
                    peaks.append(peak_candidates[i])
        
        valleys = []
        if len(valley_candidates) > 0:
            valleys.append(valley_candidates[0])
            for i in range(1, len(valley_candidates)):
                if valley_candidates[i] - valley_candidates[i-1] > 5:  # Minimum separation
                    valleys.append(valley_candidates[i])
        
        return np.array(peaks), np.array(valleys)
    
    def expand_regions(self, indices, signal_length, width):
        """Expand peak/valley regions"""
        mask = np.zeros(signal_length, dtype=bool)
        for idx in indices:
            start = max(0, idx - width)
            end = min(signal_length, idx + width + 1)
            mask[start:end] = True
        return mask
    
    def create_simple_mask(self, signal_1d):
        """Create BINARY mask - NO NORMALIZATION!"""
        # Detect peaks and valleys on RAW signal
        peaks, valleys = self.detect_peaks_valleys_simple(signal_1d)
        
        # Create base mask
        mask = np.full_like(signal_1d, self.base_weight, dtype=np.float32)
        
        # Create combined important regions
        important_regions = np.zeros(len(signal_1d), dtype=bool)
        
        # Add peak regions
        if len(peaks) > 0:
            peak_mask = self.expand_regions(peaks, len(signal_1d), self.expansion_width)
            important_regions |= peak_mask
        
        # Add valley regions
        if len(valleys) > 0:
            valley_mask = self.expand_regions(valleys, len(signal_1d), self.expansion_width)
            important_regions |= valley_mask
        
        # Apply weights
        mask[important_regions] = self.important_weight
        
        return mask, peaks, valleys
    
    def process_dataset(self, signal_data, modality_idx=1):
        """Process dataset - NO PREPROCESSING!"""
        N, channels, L = signal_data.shape
        masks = np.zeros((N, 1, L), dtype=np.float32)
        
        # Convert to numpy if tensor
        if torch.is_tensor(signal_data):
            signal_np = signal_data.numpy()
        else:
            signal_np = signal_data
        
        print(f"Creating simple masks for {N} signals (NO normalization)...")
        
        total_peaks = 0
        total_valleys = 0
        
        for i in range(N):
            if i % 100 == 0:
                print(f"Processing signal {i}/{N}")
            
            # Extract raw signal - NO PREPROCESSING!
            signal_1d = signal_np[i, modality_idx, :]
            
            # Create mask on raw signal
            mask, peaks, valleys = self.create_simple_mask(signal_1d)
            masks[i, 0, :] = mask
            
            total_peaks += len(peaks)
            total_valleys += len(valleys)
        
        print(f"Average peaks per signal: {total_peaks/N:.1f}")
        print(f"Average valleys per signal: {total_valleys/N:.1f}")
        
        return torch.tensor(masks, dtype=torch.float32)

def create_simple_masks(signal_data, save_dir="./simple_masks"):
    """Create simple masks without normalization"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Simple configuration - just percentile threshold
    mask_gen = SimpleSignalMaskGenerator(
        percentile_threshold=80,  # Top/bottom 20% of values
        expansion_width=8,
        important_weight=3.0,
        base_weight=1.0
    )
    
    # Process dataset
    masks = mask_gen.process_dataset(signal_data, modality_idx=1)
    
    # Check results
    unique_values = torch.unique(masks)
    print(f"Unique mask values: {unique_values.tolist()}")
    
    total_peak_points = (masks == 3.0).sum().item()
    total_points = masks.numel()
    peak_percentage = (total_peak_points / total_points) * 100
    print(f"Peak/Valley regions cover {peak_percentage:.1f}% of all data points")
    
    # Save masks
    filepath = os.path.join(save_dir, f'simple_masks_modality_1.pth')
    torch.save(masks, filepath)
    print(f"Simple masks saved to {filepath}")
    
    return {1: masks}, mask_gen

# Replace your current line with this:

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



#%%
    
N, channels, L = signal.shape

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
test_loader = DataLoader(test_dataset, batch_size = 1024, shuffle = True)
#%%
batch = next(iter(train_loader))
#%%
Model = CSDI_base(config, device, L = 500*4).to(device)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load("no_compress799.pth"))

train(Model,config['train'],train_loader,valid_loader=val_loader,valid_epoch_interval=10)
#%%%