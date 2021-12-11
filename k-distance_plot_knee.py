import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj 

corpus_embeddings=_load_pkl('data/Sample_embeddings.pkl') 
X = np.array(corpus_embeddings)
dim=2
pca = PCA(n_components=dim) #reduce dimension into 2
result = pca.fit_transform(X) 
neigh = NearestNeighbors(n_neighbors=dim+1)#768*2
nbrs = neigh.fit(result)
distances, indices = nbrs.kneighbors(result)
distances = np.sort(distances, axis=0)
distance_incre = distances[:,1]
from kneed import KneeLocator
kneedle = KneeLocator(range(1,len(distance_incre)+1),  #x values
                      distance_incre, # y values
                      S=1.0, #parameter suggested from paper
                      curve="convex", #parameter from figure
                      direction="increasing") #parameter from figure                      
print("kneedle.knee_y", round(kneedle.knee_y,2))
# kneedle.plot_knee_normalized()
kneedle.plot_knee() 
### full dataset
corpus_embeddings=_load_pkl('data/FVnC_dataset_embeddings.pkl') 
X = np.array(corpus_embeddings)
dim=2
pca = PCA(n_components=dim) #reduce dimension into 2
result = pca.fit_transform(X) 
neigh = NearestNeighbors(n_neighbors=dim+1)#768*2
nbrs = neigh.fit(result)
distances, indices = nbrs.kneighbors(result)
distances = np.sort(distances, axis=0)
distance_incre = distances[:,1]
from kneed import KneeLocator
kneedle = KneeLocator(range(1,len(distance_incre)+1),  #x values
                      distance_incre, # y values
                      S=1.0, #parameter suggested from paper
                      curve="convex", #parameter from figure
                      direction="increasing") #parameter from figure                      
print("kneedle.knee_y", round(kneedle.knee_y,2)) # Knee Point
# kneedle.plot_knee_normalized()
kneedle.plot_knee()  
plt.show()                 