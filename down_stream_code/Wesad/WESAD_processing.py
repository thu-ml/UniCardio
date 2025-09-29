#%%
'''
import os 
os.chdir('/root/autodl-tmp/BioDiffuser/WESAD')
import pickle
with open('S1.pkl', 'rb') as f:
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    data = u.load()
#%%
PPG = data['signal']['wrist']['BVP']   
ECG = data['signal']['chest']['ECG']   
'''
#%%
import os 
os.chdir('/root/autodl-tmp/BioDiffuser/WESAD')
import pickle
import scipy
file_name = "S19"
source_name = file_name + ".pkl"
dest_name = file_name + ".mat"
	
a=pickle.load( open( source_name, "rb" ), encoding='latin1')

scipy.io.savemat(dest_name, mdict={'pickle_data': a})