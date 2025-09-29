#%%
import scipy
import mat73
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
import numpy as np
import os
ppg = scipy.io.loadmat('ppg_segments.mat')['all_segs_ppg']
ecg = scipy.io.loadmat('ecg_segments.mat')['all_segs_ecg']
labels = scipy.io.loadmat('labels.mat')['all_labels']
all_sig = np.zeros((ppg.shape[0],2,1000))
all_sig[:,0,:] = ppg
all_sig[:,1,:] = ecg
#%%
import time

# Clean Data, using Neurokt's own built-in method (Band Pass filtering and Notch filtering). There
# are other built-in methods. Be aware where the data is recorded, the powerline noise frequency
# might need to be changed to 50 Hz.
non_nan_seg= []
labels_filt = []
s = time.time()
for trial_num in range(1,all_sig.shape[0]):
    sig = all_sig[trial_num]
    if np.isnan(sig).sum() == 0:
        sig[0,:] = nk.ecg_clean(sig[0,:], sampling_rate=125, method="neurokit",powerline = 60)
        sig[1,:] = nk.ecg_clean(sig[1,:], sampling_rate=125, method="neurokit",powerline = 60)
        
        sig[0,:]  = (sig[0,:] - np.mean(sig[0,:]))/(np.std(sig[0,:] ))
        sig[1,:]  = (sig[1,:] - np.mean(sig[1,:]))/(np.std(sig[1,:] ))
        non_nan_seg.append(sig)
        labels_filt.append(labels[trial_num])
e = time.time()
print((e - s)/60)
non_nan_seg = np.asarray(non_nan_seg)
labels_filt = np.asarray(labels_filt)


#%%
#%%
# Sample entroy can detect signals with very poor quality.
ppg_samentrop_index = []
s = time.time()
for trial_num in range(non_nan_seg.shape[0]):
    sig = non_nan_seg[trial_num][0,:]
    s,_ = nk.entropy_sample(sig)
    ppg_samentrop_index.append(s)
        
e = time.time()
print((e - s)/60)
b = np.asarray(ppg_samentrop_index)
I_ppg = np.where(b > 0.3)[0]


ecg_samentrop_index = []
s = time.time()
for trial_num in range(non_nan_seg.shape[0]):
    sig = non_nan_seg[trial_num][1,:]
    s,_ = nk.entropy_sample(sig)
    ecg_samentrop_index.append(s)
e = time.time()
print((e - s)/60)
b = np.asarray(ecg_samentrop_index)
I_ecg = np.where(b > 0.4)[0] #0.35 good try 0.3 
#%%

index_final = np.union1d(I_ppg, I_ecg)
sig_seg = np.delete(non_nan_seg, index_final, axis=0)
labels_filt = np.delete(labels_filt, index_final, axis=0)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Now first correct reverse ecg and then remove.

#%%

# Detect inverted ECG and use method from 'zhao2018' to detect low quality ecg
valid_segments = []
barely_valid_segments = []
invalid_segments = []
labels_barely = []
labels_valid = []
s = time.time()
inverted_index = 0
for index in range(sig_seg.shape[0]):
    sig = sig_seg[index,1,:]
    corrected, inv = nk.ecg_invert(sig, sampling_rate=125)
    if inv:
        sig = corrected
        sig_seg[index,1,:] = corrected
        inverted_index = inverted_index + 1
        print('Inverted', str(inverted_index))
    sig = -1 + 2*(sig - sig.min())/(sig.max() - sig.min())
    quality = nk.ecg_quality(sig, sampling_rate=125, method="zhao2018")
    if  quality == 'Unacceptable':
        invalid_segments.append(sig_seg[index,:,:])

    elif quality == 'Barely acceptable':
        barely_valid_segments.append(sig_seg[index,:,:])
        labels_barely.append(labels_filt[index])
        
    else:
        valid_segments.append(sig_seg[index,:,:])
        labels_valid.append(labels_filt[index])
e = time.time()
print((e - s)/60)

#%%    
#%%
# Min-Max normalization
final_sig = np.concatenate([np.asarray(valid_segments),np.asarray(barely_valid_segments)], axis = 0)
final_labels = np.concatenate([np.asarray(labels_valid),np.asarray(labels_barely)], axis = 0)
#%%
N, channels, L = final_sig.shape
first_half = final_sig[:, :, :L//2]
second_half = final_sig[:, :, L//2:]
final_sig = np.concatenate((first_half, second_half), axis=0)
final_labels = np.concatenate((final_labels, final_labels), axis=0)

for index in range(final_sig.shape[0]):
    final_sig[index,0,:] = -1 + 2*(final_sig[index,0,:] - final_sig[index,0,:].min())/(final_sig[index,0,:].max() - final_sig[index,0,:].min())
    final_sig[index,1,:] = -1 + 2*(final_sig[index,1,:] - final_sig[index,1,:].min())/(final_sig[index,1,:].max() - final_sig[index,1,:].min())

np.save('final_labels2.npy', final_labels)
np.save('final_sigs2.npy', final_sig)
