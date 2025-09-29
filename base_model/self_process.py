import numpy as np
import torch
def imputation_pattern(data, extended_percentage = 0.3, transient_chance = 0.3, transient_window = 5,  transient = None, extended = None):
    total_length = data.shape[2]
    mask = np.zeros_like(data)
    extended_length = int(extended_percentage*total_length)
    if extended:
        print('Adding Extended Missingness')
        for i in range(data.shape[0]):
            start_impute = np.random.randint(0, total_length - extended_length)
            data[i,:,start_impute:start_impute + extended_length] = -1
            mask[i,:,start_impute:start_impute + extended_length] = 1
    if transient:
        print("Adding Transient Missingness")
        for i in range(data.shape[0]):
            for start_impute in range(0,total_length, transient_window):
                rand = np.random.random_sample()
                if rand <= transient_chance:
                    data[i,:,start_impute : start_impute + transient_window] = -1
                    mask[i,:,start_impute : start_impute + transient_window] = 1
                    
    return torch.tensor(data), torch.tensor(mask)

def AddNoise(data, SNR):
    total_length = data.shape[2]
    for i in range(data.shape[0]):
        for k in range(data.shape[1]):
            sig_power = np.mean(data[i,k,:]**2)
            SNR_linear = 10**(SNR/10) #SNR in Linear Scale
            noise_power = sig_power/SNR_linear #Noise Power
            noise = np.random.normal(0, np.sqrt(noise_power), data[i,k,:].shape)
            data[i,k,:] = data[i,k,:] + noise
    
    return torch.tensor(data)
            