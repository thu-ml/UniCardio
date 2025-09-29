#%%
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import scipy
from scipy.signal import find_peaks, periodogram
from scipy.fftpack import fft
import sys
from self_process import imputation_pattern, AddNoise


#%%
Error_Dirty = []
Error_GEN = []
for s_id in range(2,16):
    if s_id == 8 or s_id == 12:
        continue
    else:    
        ecg_gen = np.load(f"results{s_id}_imp.npy")
        ecg_gen = np.mean(ecg_gen[:,0:100,:], axis = 1)[:,0,:]
        ecg = scipy.io.loadmat(f'Subject{s_id}_ECG.mat')['ECG']
        ecg_length = (len(ecg)//500)*500
        ecg = ecg[0:ecg_length].reshape((-1,500))
        ecg_corrupted, mask = imputation_pattern(ecg[:,None,:].copy(), extended = True)


        ecg = ecg.reshape((-1))
        ecg_corrupted = ecg_corrupted.reshape((-1))
        ecg_gen = ecg_gen.reshape((-1))
        gen_error = 0
        dirty_error = 0
        interval = 125*120
        for i in range(int(np.floor(ecg_gen.shape[0]/interval))+1):
            e = ecg_gen[interval*i:interval*(i+1)]
            e = (e - e.mean())/(e.std())
            e = -1+ 2*(e - e.min())/(e.max() - e.min())
            signals, info = nk.ecg_peaks(e, sampling_rate=125, correct_artifacts=False, show=False, method = 'nabian2018')

            gt = ecg[interval*i:interval*(i+1)]
            gt = (gt - gt.mean())/(gt.std())
            gt = -1+ 2*(gt - gt.min())/(gt.max() - gt.min())
            
            signals2, info2 = nk.ecg_peaks(gt, sampling_rate=125, correct_artifacts=False, show=False, method = 'nabian2018')
            
            p = ecg_corrupted[interval*i:interval*(i+1)]
            p = (p - p.mean())/(p.std())
            p = -1+ 2*(p - p.min())/(p.max() - p.min())
            signals3, info3 = nk.ecg_peaks(p, sampling_rate=125, correct_artifacts=False, show=False, method = 'nabian2018')

            hr_results = np.asarray(signals).sum()/(interval/125)*60
            hr_truth = np.asarray(signals2).sum()/(interval/125)*60
            hr_truth_ppg = np.asarray(signals3).sum()/(interval/125)*60

            gen_error = gen_error + np.abs(hr_results - hr_truth)
            dirty_error = dirty_error + np.abs(hr_truth_ppg - hr_truth)
        print(s_id)
        print('Gen Error: ', gen_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))
        print('Dirty Error: ', dirty_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))
        Error_GEN.append(gen_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))
        Error_Dirty.append(dirty_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))
Error_Dirty = np.asarray(Error_Dirty)
Error_GEN = np.asarray(Error_GEN)

print('Dirty Mean Error: ', Error_Dirty.mean())
print('Dirty Std Error: ', Error_Dirty.std())
print('GEN Mean Error: ', Error_GEN.mean())
print('GEN Std Error: ', Error_GEN.std())

#%%

Error_Dirty = []
Error_GEN = []
for s_id in range(2,15):
    if s_id == 8 or s_id == 12:
        continue
    else:    
        ecg_gen = np.load(f"/mnt/vepfs/audio/lichang/HeartDiffuser/Heart_diffuser/WESAD_HR/WESAD_HR/results{s_id}_de.npy")
        ecg_gen = np.mean(ecg_gen[:,0:100,:], axis = 1)[:,0,:]
        ecg = scipy.io.loadmat(f'Subject{s_id}_ECG.mat')['ECG']
        ecg_length = (len(ecg)//500)*500
        ecg = ecg[0:ecg_length].reshape((-1,500))
        # ecg_corrupted, mask = imputation_pattern(ecg[:,None,:].copy(), extended = True)
        ecg_corrupted = AddNoise(ecg[:,None,:].copy(), SNR=10)

        ecg = ecg.reshape((-1))
        ecg_corrupted = ecg_corrupted.reshape((-1))
        ecg_gen = ecg_gen.reshape((-1))
        gen_error = 0
        dirty_error = 0
        interval = 125*120
        for i in range(int(np.floor(ecg_gen.shape[0]/interval))+1):
            e = ecg_gen[interval*i:interval*(i+1)]
            e = (e - e.mean())/(e.std())
            e = -1+ 2*(e - e.min())/(e.max() - e.min())
            signals, info = nk.ecg_peaks(e, sampling_rate=125, correct_artifacts=False, show=False, method = 'nabian2018')

            gt = ecg[interval*i:interval*(i+1)]
            gt = (gt - gt.mean())/(gt.std())
            gt = -1+ 2*(gt - gt.min())/(gt.max() - gt.min())
            
            signals2, info2 = nk.ecg_peaks(gt, sampling_rate=125, correct_artifacts=False, show=False, method = 'nabian2018')
            
            p = ecg_corrupted[interval*i:interval*(i+1)]
            p = (p - p.mean())/(p.std())
            p = -1+ 2*(p - p.min())/(p.max() - p.min())
            signals3, info3 = nk.ecg_peaks(p, sampling_rate=125, correct_artifacts=False, show=False, method = 'nabian2018')

            hr_results = np.asarray(signals).sum()/(interval/125)*60
            hr_truth = np.asarray(signals2).sum()/(interval/125)*60
            hr_truth_ppg = np.asarray(signals3).sum()/(interval/125)*60

            gen_error = gen_error + np.abs(hr_results - hr_truth)
            dirty_error = dirty_error + np.abs(hr_truth_ppg - hr_truth)
        print(s_id)
        print('Gen Error: ', gen_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))
        print('Dirty Error: ', dirty_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))
        Error_GEN.append(gen_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))
        Error_Dirty.append(dirty_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))
Error_Dirty = np.asarray(Error_Dirty)
Error_GEN = np.asarray(Error_GEN)

print('Dirty Mean Error: ', Error_Dirty.mean())
print('Dirty Std Error: ', Error_Dirty.std())
print('GEN Mean Error: ', Error_GEN.mean())
print('GEN Std Error: ', Error_GEN.std())
