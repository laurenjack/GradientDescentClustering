import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

def k_gaussian_clusters(K):
    """
    Generate K Gaussian clusters on a uniform random landscape. Where each cluster has a uniform random size
    :return: 
    """
    centres = [np.random.uniform(low= -5.0, high=5.0, size=(1, 2)) for k in xrange(K)]
    #Change up the sizes of the clusters
    clust_sizes = [np.random.randint(2, 10) for k in xrange(K)]
    # clust_sizes = [5 for k in xrange(K)]
    clusters = [0.5*np.random.randn(cs, 2) + c for cs, c in zip(clust_sizes, centres)]
    return clusters, centres



# clusters, centres = k_gaussian_clusters(4)
# draw(clusters, centres)




