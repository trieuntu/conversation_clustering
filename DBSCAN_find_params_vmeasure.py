import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib
from matplotlib import rc
import csv
rc('text', usetex=True)
def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj 
#load data
corpus_embeddings=_load_pkl('../ntu_data/Sample_dataset.pkl') 
X = np.array(corpus_embeddings)
## PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X) 
labels=_load_pkl('data/label_Sample_dataset.pkl')
interval=np.arange(0.5,0.86,0.01)
header = ['eps', 'MinPts','Vscore', 'silhouette_coef', 'average', 'n_cluster']
with open('./clustering_evaluation/clustering_evaluation_pca.csv', 'w', encoding='UTF8', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  for e in interval:
    for m in range (3,6):
    ## DBSCAN
      dbscan_opt=DBSCAN(eps=e, min_samples=m)
      dbscan_opt.fit(result)
      cluster_assignment=dbscan_opt.labels_
      #metric
      Vscore=round(metrics.v_measure_score(labels,cluster_assignment),2)
      silhouette_coef=round(metrics.silhouette_score(result,cluster_assignment),2)
      avg=round((Vscore+silhouette_coef)/2,2)
      n_cluster=str(len(np.unique(cluster_assignment)))
      data=[round(e,2),m,Vscore,silhouette_coef,avg,n_cluster]
      writer.writerow(data)
      print('(eps,MinPts)('+str(round(e,2))+' ,'+str(m)+',)= Vs: ',Vscore,\
           '; sil: ',silhouette_coef,'; avg: ', avg, '; n_clus: ', n_cluster,'\n')
