#%%
# Model Loading
import scipy.io
import time
import os
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml
import sys
from diffusion_model_no_compress_final import CSDI_base
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
device = torch.device("cuda")
file_path = 'base_no_compress_original.yaml'
with open(file_path, 'r') as file:
    config = yaml.safe_load(file)

batch = torch.load('batch.pth')
Model = CSDI_base(config, device, L = 500*4).to(device)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load("no_compress799.pth"))


#%%
################################################
################################################
################################################
################################################
################################################
# PPG to ECG
Model.eval()
batch0 = batch[0][:,0,None,:]
t = time.time()
with torch.no_grad():
    results = Model(batch0.to(device), n_samples = 50, model_flag = '02', borrow_mode = 2, DDIM_flag = 0, sample_steps = 6, train_gen_flag = 1)

median_results = (results.median(dim = 1).values.detach().cpu().numpy()[:,0,:])
mean_results = (results.mean(dim = 1).detach().cpu().numpy()[:,0,:])

truth = (batch[0][:,0,1000:1500].detach().cpu().numpy())

# mask = batch[3]
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

#%%
################################################
################################################
################################################
################################################
################################################

# DDIM PPG to ECG
Model.eval()
batch0 = batch[0][:,0,None,:]
t = time.time()
with torch.no_grad():
    results = Model(batch0.to(device), n_samples = 50, model_flag = '02', borrow_mode = 2, DDIM_flag = 0, sample_steps = 6, train_gen_flag = 1)
median_results = (results[0].median(dim = 1).values.detach().cpu().numpy()[:,0,:])
mean_results = (results[0].mean(dim = 1).detach().cpu().numpy()[:,0,:])
truth = (batch[0][:,0,1000:1500].detach().cpu().numpy())

# mask = batch[3]
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
#%%
################################################
################################################
################################################
################################################
################################################
# PPG Imputation
Model.eval()
batch0 = batch[1][:,0,None,:]
t = time.time()
with torch.no_grad():
    results = Model(batch0.to(device), n_samples = 50, model_flag = '03', borrow_mode = 0, DDIM_flag = 0, sample_steps = 6, train_gen_flag = 1)

median_results = (results.median(dim = 1).values.detach().cpu().numpy()[:,0,:])
mean_results = (results.mean(dim = 1).detach().cpu().numpy()[:,0,:])

truth = (batch[0][:,0,0:500].detach().cpu().numpy())

mask = batch[3]
rmse = []
for index in range(truth.shape[0]):
    e = np.sqrt((mask[index]*(mean_results[index] - truth[index])**2).sum()/150)
    rmse.append(e)
rmse = np.array(rmse)
print("RMSE Mean: ", rmse.mean())
print("RMSE Std: ", rmse.std()) 

mae = []
for index in range(truth.shape[0]):
    e = np.absolute(mask[index]*(mean_results[index] - truth[index])).sum()/150
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
    s1 = mean_results[index, m==1]
    s2 = truth[index, m==1]
    ks_stat, ks_pvalue = analyze_ecg_distribution_difference(s1, s2)
    KS.append(ks_stat)
KS = np.array(KS)
print("KS Mean: ", KS.mean())
print("KS Std: ", KS.std())
#%%
################################################
################################################
################################################
################################################
################################################
# PPG and BP for ECG Imputation

Model.eval()
batch_input = batch[0].clone()
batch_input[:,0,1000:1500] = batch[1][:,0,1000:1500]

with torch.no_grad():
    results = Model(batch_input.to(device), n_samples = 50, model_flag = '0123', borrow_mode = 2, DDIM_flag = 0, sample_steps = 6, train_gen_flag = 1)

median_results = (results.median(dim = 1).values.detach().cpu().numpy()[:,0,:])
mean_results = (results.mean(dim = 1).detach().cpu().numpy()[:,0,:])
truth = (batch[0][:,0,1000:1500].detach().cpu().numpy())

mask = batch[3]
rmse = []
for index in range(truth.shape[0]):
    e = np.sqrt((mask[index]*(mean_results[index] - truth[index])**2).sum()/150)
    rmse.append(e)
rmse = np.array(rmse)
print("RMSE Mean: ", rmse.mean())
print("RMSE Std: ", rmse.std()) 

mae = []
for index in range(truth.shape[0]):
    e = np.absolute(mask[index]*(mean_results[index] - truth[index])).sum()/150
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
    s1 = mean_results[index, m==1]
    s2 = truth[index, m==1]
    ks_stat, ks_pvalue = analyze_ecg_distribution_difference(s1, s2)
    KS.append(ks_stat)
KS = np.array(KS)
print("KS Mean: ", KS.mean())
print("KS Std: ", KS.std())
