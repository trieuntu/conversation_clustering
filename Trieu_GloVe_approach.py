#@title Other approaches (using glove-python-binary): GloVe
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
from glove import Glove
from glove import corpus

def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj     
def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))
def make_features(v_text,stanford):
  feature=numpy.empty((len(v_text),100))
  temp=numpy.zeros((100))
  zeros_vec=temp
  index=0
  for i_text in v_text:
    i_counter=nlp.data.count_tokens(simple_tokenize(i_text))
    i_vocab = nlp.Vocab(i_counter)
    dem=0
    for word in i_vocab.idx_to_token:
      try:
        temp=temp+stanford.word_vectors[stanford.dictionary[word]][:]
      except:
        temp=temp+zeros_vec
      dem=dem+1;  
    feature[index,:]=temp/dem
    index=index+1
  return feature 

corpus=_load_pkl('./ntu_data/train_wiki.pkl')
glove=Glove()
stanford=glove.load_stanford('./glove_model/trieu_vectors.800.txt')
feature=make_features(corpus,stanford)
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