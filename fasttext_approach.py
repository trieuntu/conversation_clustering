#@title Other approaches (using GluOn): fasttext
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy 
import re
from joblib import dump
import os.path, pathlib
from sklearn.cluster import KMeans
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp

def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj     
def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))

#load data
corpus=_load_pkl('./ntu_data/train_wiki.pkl')
def create_vocab (v_text):
  s=("\n".join(v_text))
  counter = nlp.data.count_tokens(simple_tokenize(s))
  vocab = nlp.Vocab(counter)
  return counter, vocab
counter, vocab = create_vocab(corpus)
fasttext_vi = nlp.embedding.create('fasttext', source='cc.vi.300')# see more models: nlp.embedding.list_sources('FastText')
vocab.set_embedding(fasttext_vi)
def make_features(v_text):
  feature=numpy.empty((len(v_text),300))
  temp=numpy.zeros((300))
  index=0
  for i_text in v_text:
    i_counter=nlp.data.count_tokens(simple_tokenize(i_text))
    i_vocab = nlp.Vocab(i_counter)
    dem=0
    for word in i_vocab.idx_to_token:
      temp=temp+vocab.embedding[word].asnumpy()
      dem=dem+1;  
    feature[index,:]=temp/dem
    index=index+1
  return feature
feature=make_features(corpus)
print(feature.shape)
num_clusters = 5
X = feature
for dim in range(2,10):
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