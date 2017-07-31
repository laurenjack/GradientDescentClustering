import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.stats as sci

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

def k_centres_d_space(K, d, per_cluster):
    """Generate K Gaussian clusters in d space.
    - Generate the centres according to a uniform distribution [-5.0, 5.0) for each d, if
    a centre comes within a distance of 1 from another, redraw it
    - Eah Gaussian is isotropic, except for the fact the standard deviation is 0.5 rather
    than 1"""
    centres = []
    for k in xrange(K):
        new_centre = np.random.uniform(low= -5.0, high=5.0, size=(1, d))
        while is_too_close(new_centre, centres, 1.0):
            new_centre = np.random.uniform(low=-5.0, high=5.0, size=(1, d))
        centres.append(new_centre)
    clusters = [0.5*sci.truncnorm.rvs(-1.0, 1.0, scale=1.0, size=(per_cluster, 2)) + c for c in centres]
    global_opt = [np.sum((clust - c)**2.0) for clust, c in zip(clusters, centres)]
    global_opt = sum(global_opt)/float(d*K*per_cluster)
    return clusters, centres, global_opt

def is_too_close(new_centre, centres, dist):
    for c in centres:
        if np.linalg.norm(new_centre - c) < dist:
            return True
    return False



# clusters, centres = k_gaussian_clusters(4)
# draw(clusters, centres)




