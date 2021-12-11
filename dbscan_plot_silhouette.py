from operator import le
import pickle
from numpy.random import RandomState
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import plotly.express as px
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'Latin Modern Math'

def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj 
#load data
embeddings=_load_pkl('data/Sample_embeddings.pkl')
corpus_embeddings=embeddings
##PCA
X = np.array(corpus_embeddings)
pca = PCA(n_components=2, random_state=1) #reduce dimension into 2
result = pca.fit_transform(X) 
## DBSCAN
dbscan_opt=DBSCAN(eps=0.84,min_samples=4)
dbscan_opt.fit(result)
cluster_assignment=dbscan_opt.labels_
#calculate noise
no_noise = np.sum(np.array(cluster_assignment) == -1, axis=0)
colors = ['red','green','blue','magenta','black']
colors_remove_nosie = ['green','blue','magenta','black']
########remove noise#########
label_remove_noise=[]
range_max = len(result)
label_remove_noise = np.array([cluster_assignment[i] for i in range(0, range_max) if cluster_assignment[i] != -1])
X_remove_noise = np.array([result[i] for i in range(0, range_max) if cluster_assignment[i] != -1])
##
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(5.8, 2.2)
##scatter 1
scatter1 = ax1.scatter(result[:, 0], result[:, 1],c=cluster_assignment,cmap=matplotlib.colors.ListedColormap(colors),s=15)
legend1 = ax1.legend(*scatter1.legend_elements(),loc="upper right", title="Cluster",prop={'size': 6})
ax1.add_artist(legend1)
ax1.set_title('(a)')
##scatter 2
scatter2 = ax2.scatter(X_remove_noise[:, 0], X_remove_noise[:, 1],c=label_remove_noise,\
          cmap=matplotlib.colors.ListedColormap(colors_remove_nosie),s=15)
legend2 = ax2.legend(*scatter2.legend_elements(), prop={'size': 7}, loc="upper left", title="Cluster")
ax2.add_artist(legend2)
ax2.set_title('(b)')
### silhouette plot
range_n_clusters = [4]
X=X_remove_noise
colorss=[]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots()
    # fig.set_size_inches(5.8, 2.2)
    fig.set_size_inches(5.9/2, 2.3)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    # cluster_labels = clusterer.fit_predict(X)
    cluster_labels=label_remove_noise

    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color=colors_remove_nosie[i]
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([]) 
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()
