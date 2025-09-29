#%%
import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
#%%
sampling_rate = 500

Y = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
Y.head()

def load_raw_data(df, sr):
    if sr == 100:
        data = [wfdb.rdsamp(f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
X = load_raw_data(Y, sampling_rate)
#%%
agg_df = pd.read_csv('scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]
agg_df.head()
# %%
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
#%%
y = Y.diagnostic_superclass
y = y.to_numpy()

def remove_empty(X, y):
    empty_indexes = []
    for i, v in enumerate(y):
        if len(v) == 0:
            empty_indexes.append(i)

    X = np.delete(X, empty_indexes, axis=0)
    y = np.delete(y, empty_indexes)
    
    return X, y

X, y = remove_empty(X, y)

get_first_class = lambda t: t[0]
vfunc = np.vectorize(get_first_class)
y = vfunc(y)

def class_to_int(x):
    if x == 'NORM':
        return 0
    elif x == 'CD':
        return 1
    elif x == 'MI':
        return 2
    elif x == 'HYP':
        return 3
    elif x == 'STTC':
        return 4

y = np.array(list(map(class_to_int, y)))
#%%
from scipy import signal
import neurokit2 as nk

xII = X[:,:,1]
xII = signal.decimate(xII, q=4)
for i in range(xII.shape[0]):
    xII[i,:] = nk.ecg_clean(xII[i,:], sampling_rate=125, method="neurokit",powerline = 50)
    xII[i,:] = (xII[i,:] - np.mean(xII[i,:]))/(np.std(xII[i,:]))



valid_segments = []
barely_valid_segments = []
invalid_segments = []


valid_labels = []
barely_valid_labels = []
invalid_labels = []

inverted_index = 0
for index in range(xII.shape[0]):
    sig = xII[index,:]
    label = y[index]
    corrected, inv = nk.ecg_invert(sig, sampling_rate=125)
    if inv:
        sig = corrected
        xII[index,:] = corrected
        print('Inverted', str(inverted_index))
    sig = -1 + 2*(sig - sig.min())/(sig.max() - sig.min())

    quality = nk.ecg_quality(sig, sampling_rate=125, method="zhao2018")
    if  quality == 'Unacceptable':
        invalid_segments.append(xII[index,:])
        invalid_labels.append(label)

    elif quality == 'Barely acceptable':
        barely_valid_segments.append(xII[index,:])
        barely_valid_labels.append(label)
        
    else:
        valid_segments.append(xII[index,:])
        valid_labels.append(label)
    
final_sig = np.concatenate([np.asarray(valid_segments),np.asarray(barely_valid_segments)], axis = 0)
final_y = np.concatenate([np.asarray(valid_labels),np.asarray(barely_valid_labels)], axis = 0)

#%%
import torch
xII_part1 = final_sig[:,0:500]
xII_part2 = final_sig[:,500:1000]
x_II_seg = np.concatenate([xII_part1, xII_part2], axis=0)
label_seg = np.concatenate([final_y, final_y], axis=0)

for i in range(x_II_seg.shape[0]):
    x_II_seg[i,:] = -1 + 2*(x_II_seg[i,:] - x_II_seg[i,:].min())/(x_II_seg[i,:].max() - x_II_seg[i,:].min())
#%%
np.save('x_II_seg.npy', x_II_seg)
np.save('label_seg.npy', label_seg)

