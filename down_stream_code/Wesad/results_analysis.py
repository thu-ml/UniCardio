#%%
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import scipy
from scipy.signal import find_peaks, periodogram
from scipy.fftpack import fft

def safe_ppg_peaks(signal, sampling_rate=125):
    """
    A safer wrapper around nk.ppg_peaks that handles potential failures
    Returns (peaks_array, info) if successful, or (None, None) if detection fails
    """
    try:
        # First verify we have enough data for meaningful peak detection
        if len(signal) < sampling_rate:  # Need at least 1 second
            return None, None
            
        # Check if signal has meaningful variation
        if np.std(signal) < 1e-10:  # Almost flat signal
            return None, None
            
        # Try alternative peak detection methods if the default fails
        try:
            signals, info = nk.ppg_peaks(signal, sampling_rate=sampling_rate, 
                                       method="elgendi",  # First try default
                                       correct_artifacts=False, 
                                       show=False)
        except IndexError:
            # If Elgendi method fails, try Bishop method
            try:
                signals, info = nk.ppg_peaks(signal, sampling_rate=sampling_rate,
                                           method="bishop",  # Alternative method
                                           correct_artifacts=False,
                                           show=False)
            except:
                return None, None
                
        return signals, info
        
    except Exception as e:
        print(f"Peak detection failed: {str(e)}")
        return None, None


Error_PPG = []
Error_GEN = []
#2, 18
for s_id in range(2,18):
    if s_id == 8 or s_id == 12:
        continue
    else:    
        ecg_gen = np.load(f'results{s_id}.npy')
        ecg_gen = np.mean(ecg_gen[:,0:100,:], axis = 1)[:,0,:].reshape((-1))

        ppg = scipy.io.loadmat(f'Subject{s_id}_PPG.mat')['PPG']
        ecg = scipy.io.loadmat(f'Subject{s_id}_ECG.mat')['ECG']

        ppg = ppg[0,0:ecg_gen.shape[0]]
        ecg = ecg[0:ecg_gen.shape[0],0]

        length = int(np.floor(ecg_gen.shape[0]/500))
        ppg = ppg[0:500*length].reshape((length, 500))
        ecg = ecg[0:500*length].reshape((length, 500))
        ecg_gen = ecg_gen[0:500*length].reshape((length, 500))

        ecg = ecg.reshape((-1))
        ppg = ppg.reshape((-1))
        ecg_gen = ecg_gen.reshape((-1))
        gen_error = 0
        ppg_error = 0
        interval = 125*120
        for i in range(int(np.floor(ecg_gen.shape[0]/interval))):
            e = ecg_gen[interval*i:interval*(i+1)]
            e = (e - e.mean())/(e.std())
            e = -1+ 2*(e - e.min())/(e.max() - e.min())

            gt = ecg[interval*i:interval*(i+1)]
            gt = (gt - gt.mean())/(gt.std())
            gt = -1+ 2*(gt - gt.min())/(gt.max() - gt.min())

            signals, info = nk.ecg_peaks(e, sampling_rate=125, correct_artifacts=False, show=False, method = 'nabian2018')
            signals2, info2 = nk.ecg_peaks(gt, sampling_rate=125, correct_artifacts=False, show=False, method = 'nabian2018')
            p = ppg[interval*i:interval*(i+1)]
            p = (p - p.mean())/(p.std())
            p = -1+ 2*(p - p.min())/(p.max() - p.min())
            signals3, info3 =  safe_ppg_peaks(p, sampling_rate=125)
            hr_results = np.asarray(signals).sum()/(interval/125)*60
            hr_truth = np.asarray(signals2).sum()/(interval/125)*60
            if signals3 is not None:
                hr_truth_ppg = np.asarray(signals3).sum()/(interval/125)*60
            else:
                print(f"Skipping segment {i} due to failed peak detection")
                hr_truth_ppg = 0
                continue

            gen_error = gen_error + np.abs(hr_results - hr_truth)
            ppg_error = ppg_error + np.abs(hr_truth_ppg - hr_truth)
        print(s_id)
        print('Gen Error: ', gen_error/(int(np.floor(ecg_gen.shape[0]/interval))))
        print('PPG_error: ', ppg_error/(int(np.floor(ecg_gen.shape[0]/interval))))
        Error_GEN.append(gen_error/(int(np.floor(ecg_gen.shape[0]/interval))))
        Error_PPG.append(ppg_error/(int(np.floor(ecg_gen.shape[0]/interval))))
Error_PPG = np.asarray(Error_PPG)
Error_GEN = np.asarray(Error_GEN)
print('PPG Mean Error: ', Error_PPG.mean())
print('PPG Std Error: ', Error_PPG.std())
print('GEN Mean Error: ', Error_GEN.mean())
print('GEN Std Error: ', Error_GEN.std())

#%%

signals2, info2 = nk.ecg_peaks(e, sampling_rate=125, correct_artifacts=False, show=True, method = 'neurokit')
signals3, info3 = nk.ppg_peaks(p, sampling_rate=125, correct_artifacts=False, show=True)
signals3, info3 = nk.ecg_peaks(ecg[interval*i:interval*(i+1)], sampling_rate=125, correct_artifacts=False,method="neurokit", show=True)



#%%
gen_error = 0
ppg_error = 0
interval = 125*600
for i in range(int(np.floor(ecg_gen.shape[0]/interval))+1):
    e = ecg_gen[interval*i:interval*(i+1)]
    e = (e - e.mean())/(e.std())
    e = -1+ 2*(e - e.min())/(e.max() - e.min())
    signals, info = nk.ecg_peaks(e, sampling_rate=125, correct_artifacts=False, show=False, method = 'neurokit')

    signals2, info2 = nk.ecg_peaks(ecg[interval*i:interval*(i+1)], sampling_rate=125, correct_artifacts=False, show=False, method = 'neurokit')

    p = ppg[interval*i:interval*(i+1)]
    p = (p - p.mean())/(p.std())
    p = -1+ 2*(p - p.min())/(p.max() - p.min())
    signals3, info3 = nk.ppg_peaks(p, sampling_rate=125, correct_artifacts=True, show=False)

    rate1 = nk.signal_rate(signals,
                    desired_length=len(signals),
                    sampling_rate = 125,
                    interpolation_method="nearest")

    rate2 = nk.signal_rate(signals2,
                        desired_length=len(signals),
                        sampling_rate = 125,
                        interpolation_method="nearest")

    rate3 = nk.signal_rate(signals3,
                        desired_length=len(signals3),
                        sampling_rate = 125,
                        interpolation_method="nearest")

    gen_error = gen_error + np.abs(rate1 -rate2).mean()
    ppg_error = ppg_error + np.abs(rate3 -rate2).mean()
print('Gen Error: ', gen_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))
print('PPG_error: ', ppg_error/(int(np.floor(ecg_gen.shape[0]/interval))+1))

#%%
gen_error = 0
ppg_error = 0
signals, info = nk.ecg_peaks(ecg_gen, sampling_rate=125, correct_artifacts=False, show=False, method = 'neurokit')
signals2, info2 = nk.ecg_peaks(ecg, sampling_rate=125, correct_artifacts=False, show=False, method = 'neurokit')
signals3, info3 = nk.ppg_peaks(ppg, sampling_rate=125, correct_artifacts=False,method="elgendi", show=False)

rate1 = nk.signal_rate(signals,
                    desired_length=len(signals),
                    sampling_rate = 125,
                    interpolation_method="nearest")

rate2 = nk.signal_rate(signals2,
                    desired_length=len(signals),
                    sampling_rate = 125,
                    interpolation_method="nearest")

rate3 = nk.signal_rate(signals3,
                    desired_length=len(signals3),
                    sampling_rate = 125,
                    interpolation_method="nearest")

gen_error = gen_error + np.abs(rate1 -rate2).mean()
ppg_error = ppg_error + np.abs(rate3 -rate2).mean()
print('Gen Error: ', gen_error)
print('PPG_error: ', ppg_error)
