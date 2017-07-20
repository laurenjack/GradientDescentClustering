from cluster_gen import k_gaussian_clusters
from grad_desc_clustering import GDC
from centre_animator import animate
import k_means
import numpy as np


K = 4
lr = 2.0
epochs = 60
#Generate the clusters
clusters, actual_centres = k_gaussian_clusters(K)
#X = [x.transpose() for x in clusters]
X = np.concatenate(clusters, axis=0)
#X = X.transpose()

#Create same set of starting centres
X_bar = np.random.uniform(low= -5.0, high=5.0, size=(K, 2))

#Train the clustering algorithm
gdc = GDC()
W, X_bar_gdc, all_prev_gdc = gdc.train(X, K, lr, epochs, X_bar)

X_bar_kMeans, all_prev_kMeans = k_means.train(X, K, epochs, X_bar)
all_prev_lists = [all_prev_gdc, all_prev_kMeans]

#Animate
animate(clusters, all_prev_lists)

# n_div_k,  _, d = clusters.shape
# n = n_div_k * K
# X = clusters.transpose()
# X = X.reshape(d, n)
# X = X.transpose()