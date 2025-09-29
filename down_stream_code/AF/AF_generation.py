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

batch = torch.load('batch.pth')
Model = CSDI_base(config, device, L = 500*4)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load('no_compress799.pth'))


all_sig = np.load('final_sigs2.npy')
all_sig = torch.tensor(all_sig,dtype=torch.float32)

null = torch.zeros([all_sig.shape[0], int(all_sig.shape[2])])
all_sig = torch.concatenate([all_sig[:,0,:], null, all_sig[:,1,:], null], dim = -1)

all_labels = np.load('final_labels2.npy')


from torch.utils.data import Dataset
class CustomSignalDataset(Dataset):
    def __init__(self, signal,all_labels):
        self.signal = signal
        self.all_labels = all_labels

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, idx):
        return self.signal[idx, None, :], self.all_labels[idx, :]

val_dataset = CustomSignalDataset(all_sig,all_labels)
val_loader = DataLoader(val_dataset, batch_size = 512, shuffle = False)
#%%
print('second_run')
# Set the model to evaluation mode
Model.eval()

# Directory to save the results
os.makedirs('inference_results', exist_ok=True)

# Loop through the DataLoader and perform inference on each batch
for i, batch in enumerate(val_loader):
    input_sig = batch[0].clone()
    start = time.time()
    results = Model(input_sig.to(device), 50, '02', borrow_mode=0)
    end = time.time()

    # Save the results for each batch
    results_path = f'inference_results/results_batch_large_{i}.pt'
    torch.save(results, results_path)
    
    results_path = f'inference_results/input_batch_large{i}.pt'
    torch.save(input_sig, results_path)
    
    results_path = f'inference_results/labels_batch_large{i}.pt'
    torch.save( batch[1].clone(), results_path)
    

    # Clear memory
    del input_sig, results
    torch.cuda.empty_cache()
    
    print(f'Batch {i} processed in {(end - start) / 60:.2f} minutes. Results saved to {results_path}')
