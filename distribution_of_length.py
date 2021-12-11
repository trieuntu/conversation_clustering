import pandas as pd
import pickle
from pyvi import ViTokenizer, ViPosTagger
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Latin Modern Math'
from matplotlib import rc
rc('text', usetex=True)

def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj 
train=_load_pkl('data/FVnC_dataset.pkl')#DLU_without_coincide_text_train
print(len(train))  
seq_len = [len(i.split()) for i in train]
ts=pd.Series(seq_len).hist(color="b", bins=300)
ts.plot()
ts.set_xlim((4,65))
ts.grid(False)
plt.show()