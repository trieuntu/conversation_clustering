#@title Difference models: gpt2, distilbert-base-multilingual-cased, vinai/phobert-base, bert-base-uncased, bert-base-multilingual-uncased { form-width: "10%" }
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy
import re
from transformers import AutoModel, AutoTokenizer 
from joblib import dump
from transformers import BertModel, BertConfig
from sklearn.cluster import KMeans
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

def load_bert():
    v_phobert = AutoModel.from_pretrained('bert-base-multilingual-uncased',output_hidden_states=True)
    v_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', use_fast=False)
    return v_phobert, v_tokenizer
# Taking features from BERT Model
def get_CLS_embedding(layer):
    return layer[:, 0, :].numpy()
def make_bert_features(v_text):
    global phobert, tokenizer
    v_tokenized = []
    max_len = 255
    for i_text in v_text:
        line = tokenizer.encode(i_text, truncation=True, max_length=max_len)
        v_tokenized.append(line)
    padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
    attention_mask = numpy.where(padded == 1, 0, 1)
    padded = torch.tensor(padded).to(torch.long)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = phobert(input_ids= padded, attention_mask=attention_mask)
    v_features = last_hidden_states[0][:, 0, :].numpy()
    return v_features
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj 

print("Loading new model....")
phobert, tokenizer = load_bert()
print("Finished loading model!")
#load data
corpus=_load_pkl('./ntu_data/train_wiki.pkl')
lines=corpus  
print("Preparing to create features .....")
## define interval
interval=50
corpus_embeddings = make_bert_features(lines[:interval])
print("done interval:",0,"-",interval)
for i in range (int(len(lines)/interval)-1):
    j=(i+1)*interval
    k=j+interval
    corpus_embeddings=numpy.concatenate((corpus_embeddings,make_bert_features(lines[j:k])),axis=0)
    print("done interval:",j,"-",k)
if(k<len(lines)):  
    print("last interval:")  
    corpus_embeddings=numpy.concatenate((corpus_embeddings,make_bert_features(lines[k:len(lines)])),axis=0)

print("Finished creating features from BERT")
#######################################
num_clusters = 5
X = numpy.array(corpus_embeddings)
for dim in range(2,150):
  pca = PCA(n_components=dim)
  result = pca.fit_transform(X) 
  clustering_model = KMeans(n_clusters=num_clusters,random_state=1)
  clustering_model.fit(result)
  cluster_assignment = clustering_model.labels_
  labels=_load_pkl('./ntu_data/label_wiki.pkl')
  Vscore=round(metrics.v_measure_score(labels,cluster_assignment),2)
  silhouette_coef=round(metrics.silhouette_score(result,cluster_assignment),2)
  print('V Score= ',Vscore, ' Silhouette Score= ', silhouette_coef, \
        ' dim=',dim, '\n')
