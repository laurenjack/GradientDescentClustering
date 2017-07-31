from cluster_gen import *
from grad_desc_clustering import GDC
from convex_opt import *
import k_means
import numpy as np
import cluster_utils
#from centre_animator import animate



K = 10
d = 4
lr = 0.18
epochs = 200

total_opt = 0
total_gdc = 0
total_kMeans = 0
gdc_under = 0
kMeans_under = 0

for i in xrange(100):
    # Generate the clusters
    clusters, actual_centres, global_opt = k_centres_d_space(K, d, 5)
    X = np.concatenate(clusters, axis=0)

    # Choose K for GDC
    gdc = GDC()
    # co = ConvexOptimizer(gdc)
    # tp = TrainingParams(X, lr, epochs)
    # K, _ = co.bin_search(tp, 0, 10)

    # Create same set of starting centres
    X_bar = np.random.uniform(low=-5.0, high=5.0, size=(K, d))

    # Train the clustering algorithms
    W, X_bar_gdc, all_prev_gdc = gdc.train(X, K, lr, epochs, X_bar)

    kMeans = k_means.KMeans()
    X_bar_kMeans, all_prev_kMeans = kMeans.train(X, K, epochs, X_bar)

    # Compute total cost of each clustering alg
    gdc_C = cluster_utils.cost_of_closest_to(X, X_bar_gdc)
    kMeans_C = cluster_utils.cost_of_closest_to(X, X_bar_kMeans)

    #Update agregates
    total_opt += global_opt
    total_gdc += gdc_C
    total_kMeans += kMeans_C
    if gdc_C < global_opt:
        gdc_under += 1
    if kMeans_C < global_opt:
        kMeans_under += 1

print "Global Optimum: "+str(total_opt/100.0)
print "GDC: "+str(total_gdc/100.0)+"  Percent Under: "+str(gdc_under)
print "K Means: "+str(total_kMeans/100.0)+"  Percent Under: "+str(kMeans_under)


# print "Global Optimum: "+str(global_opt)
# print "GDC: "+str(gdc_C)
# print "K Means: "+str(kMeans_C)