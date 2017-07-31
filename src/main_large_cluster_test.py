from cluster_gen import *
from grad_desc_clustering import GDC
from convex_opt import *
from centre_animator import animate
import k_means
import numpy as np
import cluster_utils

K = 4
d = 2
lr = 4.0
epochs = 200
# Generate the clusters
clusters, actual_centres, global_opt = k_centres_d_space(K, d, 5)
X = np.concatenate(clusters, axis=0)

# Choose K for GDC
gdc = GDC()
# co = ConvexOptimizer(gdc)
# tp = TrainingParams(X, lr, epochs)
# K, _ = co.bin_search(tp, 0, 10)

# Create same set of starting centres
X_bar = np.random.uniform(low= -5.0, high=5.0, size=(K, d))

# Train the clustering algorithms
W, X_bar_gdc, _ = gdc.train(X, K, lr, epochs, X_bar)

kMeans = k_means.KMeans()
X_bar_kMeans, _ = kMeans.train(X, K, epochs, X_bar)

#Compute total cost of each clustering alg
gdc_C = cluster_utils.cost_of_closest_to(X, X_bar_gdc)
kMeans_C = cluster_utils.cost_of_closest_to(X, X_bar_kMeans)
print "Global Optimum: "+str(global_opt)
print "GDC: "+str(gdc_C)
print "K Means: "+str(kMeans_C)