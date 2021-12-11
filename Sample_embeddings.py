import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy
import re
from transformers import AutoModel, AutoTokenizer # Thư viện BERT
from joblib import dump
from transformers import BertModel, BertConfig
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
corpus=_load_pkl('data/Sample_dataset.pkl') 
lines=corpus  # take all sentences
print("Preparing to create features from BERT.....")
cls_embeddings = []
## define interval
interval=50
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
_save_pkl('Sample_embeddings.pkl', corpus_embeddings)    #remove coincide
print("Finished creating features from BERT")
#######################################
X = numpy.array(corpus_embeddings)
pca = PCA(n_components=2)
result = pca.fit_transform(X) 
#number of clusters
num_clusters = 3
# Define kmeans model
clustering_model = KMeans(n_clusters=num_clusters,random_state=1)
# Fit the embedding with kmeans clustering.
clustering_model.fit(result)
cluster_assignment = clustering_model.labels_
labels=_load_pkl('data/label_Sample_dataset.pkl')
print(metrics.v_measure_score(labels,cluster_assignment)*100)
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(lines[sentence_id])
#save records    
new_dir_name = 'OUTPUT'
new_dir = pathlib.Path('./', new_dir_name)
new_sub_dir_name = str(num_clusters)+' clusters'
new_sub_dir=pathlib.Path(new_dir,new_sub_dir_name)
for i, cluster in enumerate(clustered_sentences):
    name_of_file= "Cluster "+str(i+1)+".txt"
    txtlogPath = os.path.join(new_sub_dir, name_of_file)
    os.makedirs(os.path.dirname(txtlogPath), exist_ok=True)
    with open(txtlogPath, 'w') as f:
      count=0
      for item in cluster:
        count+=1
        f.write("%s\n\n####################Conversation %s####################\n\n" % (item,str(count)))
    print("done cluster",i)  
# Plotting the resulting clusters
fig, ax = plt.subplots()
scatter = ax.scatter(result[:, 0], result[:, 1],c=cluster_assignment,s=16)
legend = ax.legend(*scatter.legend_elements(),loc="lower right", title="Cluster")
ax.add_artist(legend)
plt.show()
