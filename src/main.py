from cluster_gen import *
from grad_desc_clustering import GDC
from convex_opt import *
from centre_animator import animate
import k_means
import numpy as np
import cluster_utils

K = 3
per_clust = 4
lr = 1.0
d = 1
epochs = 400
# Generate the clusters
#clusters, actual_centres, global_opt = k_gaussian_clusters(K)
# clusters, actual_centres, global_opt = k_centres_d_space(K, 2, 10)
# X = np.concatenate(clusters, axis=0)
#X = uniform_field()
X = K_clust_on_line(K, per_clust, ep=-0.0)
#X = np.array([-6.875, -5.625, -4.375]).reshape(3, 1) #, 4.375, 5.625, 6.875]).reshape(6, 1)

# Choose K for GDC
gdc = GDC()
# co = ConvexOptimizer(gdc)
# tp = TrainingParams(X, lr, epochs)
# K, _ = co.bin_search(tp, 0, 10)

# Create same set of starting centres
#X_bar = np.random.uniform(low=-10, high=10, size=(K, d))
#X_bar = np.random.uniform(low=-5.0, high=5.0, size=(K, d))
#X_bar = np.array([-5.0, 0.0, 5.0]).reshape(K, 1)
X_bar = np.array([-4.375, 1.25, 5.625]).reshape(K, 1)
#X_bar = np.array([-6.0]).reshape(K, 1) # , 5.6252
#X_bar = np.array([-2.5, 3.75, 6.25]).reshape(K, 1)
#X_bar = np.array([-5.0, -1.667, 1.667, 5.0]).reshape(4, 1)
#START
print X_bar


# Train the clustering algorithms
#p, X_bar_gdc, all_prev_gdc, all_grads = gdc.train_groups(X, 8, lr, epochs, X_bar)
#p, X_bar_gdc, all_prev_gdc, all_grads = gdc.train(X, K, lr, epochs, X_bar, L=None)
p, X_bar_gdc, all_prev_gdc = gdc.train_sgd(X, K, lr, epochs, 12, X_bar)
#p_w, X_bar_gdc_w, all_prev_gdc_weights = gdc.train_sgd(X, K, lr, epochs, 10, X_bar, weights=True)

#kMeans = k_means.KMeans()
#X_bar_kMeans, all_prev_kMeans = kMeans.train(X, K, epochs, X_bar)
#all_prev_lists = [all_prev_gdc, all_prev_kMeans]




#Compute total cost of each clustering alg
gdc_C = cluster_utils.cost_of_closest_to(X, X_bar_gdc)
#gdc_C_w = cluster_utils.cost_of_closest_to(X, X_bar_gdc_w)
#kMeans_C = cluster_utils.cost_of_closest_to(X, X_bar_kMeans)
#print "Global Optimum: "+str(global_opt)
print "GDC: "+str(gdc_C)
#print "GDC W: "+str(gdc_C_w)
#print "X_bar: "+str(X_bar_gdc)
#print "p : "+str(p)

# Make X 2D for display
def make_2D(arr, num_elm):
    return np.array([arr[:, 0], np.zeros(num_elm)]).transpose()
X = make_2D(X, K*per_clust)
all_prev_gdc = [make_2D(clusters, K) for clusters in all_prev_gdc]

all_prev_lists = [all_prev_gdc]#, all_prev_gdc_weights]

# Animate
animate([X], all_prev_lists)
# animate(clusters, all_prev_lists)

# n_div_k,  _, d = clusters.shape
# n = n_div_k * K
# X = clusters.transpose()
# X = X.reshape(d, n)
# X = X.transpose()
