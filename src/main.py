from cluster_gen import k_gaussian_clusters
from grad_desc_clustering import GDC
from convex_opt import *
from centre_animator import animate
import k_means
import numpy as np

K = 4
lr = 4.0
epochs = 120
# Generate the clusters
clusters, actual_centres = k_gaussian_clusters(K)
X = np.concatenate(clusters, axis=0)

# Choose K for GDC
gdc = GDC()
# co = ConvexOptimizer(gdc)
# tp = TrainingParams(X, lr, epochs)
# K, _ = co.bin_search(tp, 0, 10)

# Create same set of starting centres
X_bar = np.random.uniform(low= -5.0, high=5.0, size=(K, 2))

# Train the clustering algorithms
W, X_bar_gdc, all_prev_gdc = gdc.train(X, K, lr, epochs, X_bar)

kMeans = k_means.KMeans()
X_bar_kMeans, all_prev_kMeans = kMeans.train(X, K, epochs, X_bar)
all_prev_lists = [all_prev_gdc, all_prev_kMeans]

# Animate
animate(clusters, all_prev_lists)

# n_div_k,  _, d = clusters.shape
# n = n_div_k * K
# X = clusters.transpose()
# X = X.reshape(d, n)
# X = X.transpose()
