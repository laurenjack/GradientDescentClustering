from cluster_gen import *
from grad_desc_clustering import GDC
from convex_opt import *
from centre_animator import animate
import k_means
import numpy as np
import cluster_utils

K = 16
lr =  1.4
epochs = 400
# Generate the clusters
#clusters, actual_centres, global_opt = k_gaussian_clusters(K)
# clusters, actual_centres, global_opt = k_centres_d_space(K, 2, 10)
# X = np.concatenate(clusters, axis=0)
X = uniform_field()

# Choose K for GDC
gdc = GDC()
# co = ConvexOptimizer(gdc)
# tp = TrainingParams(X, lr, epochs)
# K, _ = co.bin_search(tp, 0, 10)

# Create same set of starting centres
X_bar = np.random.uniform(low= -5.0, high=5.0, size=(K, 2))

# Train the clustering algorithms
#p, X_bar_gdc, all_prev_gdc, all_grads = gdc.train_groups(X, 8, lr, epochs, X_bar)
#p, X_bar_gdc, all_prev_gdc, all_grads = gdc.train(X, K, lr, epochs, X_bar, L=None)
p, X_bar_gdc, all_prev_gdc = gdc.train_sgd(X, K, lr, epochs, 256, X_bar)
#p_w, X_bar_gdc_w, all_prev_gdc_weights = gdc.train_sgd(X, K, lr, epochs, 10, X_bar, weights=True)

#kMeans = k_means.KMeans()
#X_bar_kMeans, all_prev_kMeans = kMeans.train(X, K, epochs, X_bar)
#all_prev_lists = [all_prev_gdc, all_prev_kMeans]
all_prev_lists = [all_prev_gdc]#, all_prev_gdc_weights]



#Compute total cost of each clustering alg
gdc_C = cluster_utils.cost_of_closest_to(X, X_bar_gdc)
#gdc_C_w = cluster_utils.cost_of_closest_to(X, X_bar_gdc_w)
#kMeans_C = cluster_utils.cost_of_closest_to(X, X_bar_kMeans)
#print "Global Optimum: "+str(global_opt)
print "GDC: "+str(gdc_C)
#print "GDC W: "+str(gdc_C_w)
#print "X_bar: "+str(X_bar_gdc)
#print "p : "+str(p)

# Animate
animate([X], all_prev_lists)
# animate(clusters, all_prev_lists)

# n_div_k,  _, d = clusters.shape
# n = n_div_k * K
# X = clusters.transpose()
# X = X.reshape(d, n)
# X = X.transpose()
