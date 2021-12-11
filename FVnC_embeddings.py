import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy
from transformers import AutoModel, AutoTokenizer 
from joblib import dump
from transformers import BertModel, BertConfig
import os.path, pathlib
import pickle
from sklearn.cluster import KMeans



# save function
def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

# function for load Pretrained PhoBERT model
def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base",output_hidden_states=True)
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

# Taking features from BERT Model
def get_CLS_embedding(layer):
    return layer[:, 0, :].numpy()
def make_bert_features(v_text):
    global phobert, tokenizer
    v_tokenized = []
    max_len = 33 # max length of each sentence is 33 from histogram
    for i_text in v_text:
        # Tokenize by BERT
        line = tokenizer.encode(i_text, truncation=True, max_length=max_len)
        v_tokenized.append(line)
    padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
    attention_mask = numpy.where(padded == 1, 0, 1)
    # transform to tensor
    padded = torch.tensor(padded).to(torch.long)
    attention_mask = torch.tensor(attention_mask)
    # take features from 12th layer
    with torch.no_grad():
        last_hidden_states = phobert(input_ids= padded, attention_mask=attention_mask)
    v_features = last_hidden_states[0][:, 0, :].numpy()
    return v_features
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj 

print("Loading PhoBERT model....")
phobert, tokenizer = load_bert()
print("Finished loading model!")
#load data
corpus=_load_pkl('./ntu_data/DLU_without_coincide_text_train.pkl') 
lines=corpus  # take all sentences
print("Preparing to create features from BERT.....")
cls_embeddings = []
## define interval
interval=500
##use bert for extracting features
corpus_embeddings = make_bert_features(lines[:interval])
for i in range (int(len(lines)/interval)-1):
    j=(i+1)*interval
    k=j+interval
    corpus_embeddings=numpy.concatenate((corpus_embeddings,make_bert_features(lines[j:k])),axis=0)
    print("done interval:",j,"-",k)
if(k<len(lines)):  
    print("last interval:")  
    corpus_embeddings=numpy.concatenate((corpus_embeddings,make_bert_features(lines[k:len(lines)])),axis=0)
_save_pkl('FVnC_dataset_embeddings.pkl', corpus_embeddings)    #remove coincide
print("Finished creating features from BERT")