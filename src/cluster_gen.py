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
    n = sum(clust_sizes)
    global_opt = _compute_global_opt(clusters, centres, 2, n)
    return clusters, centres, global_opt

def k_centres_d_space(K, d, per_cluster):
    """Generate K Gaussian clusters in d space.
    - Generate the centres according to a uniform distribution [-5.0, 5.0) for each d, if
    a centre comes within a distance of 2 from another, redraw it
    - Eah Gaussian is isotropic, except for the fact the standard deviation is 0.5 rather
    than 1"""
    centres = []
    for k in xrange(K):
        new_centre = np.random.uniform(low= -5.0, high=5.0, size=(1, d))
        while is_too_close(new_centre, centres, 2.0):
            new_centre = np.random.uniform(low=-5.0, high=5.0, size=(1, d))
        centres.append(new_centre)
        #one_dist = np.random.randn(per_cluster, d)
        # one_dist = sci.truncnorm.rvs(-1.0, 1.0, scale=1.0, size=(per_cluster, d))
    pcs = [per_cluster for i in xrange(K)] # + np.random.randint(-5, 6)
    clusters = [2.0 * np.random.uniform(size=(pc, d)) + c for c, pc in zip(centres, pcs)] #0.5 * np.random.randn(pc, d)
    global_opt = _compute_global_opt(clusters, centres, d, sum(pcs))
    return clusters, centres, global_opt

def uniform_field():
    coord_grid = np.meshgrid(np.arange(32, dtype=np.float32), np.arange(32, dtype=np.float32))
    return 0.75 * np.array([xi.flatten() for xi in coord_grid]).transpose() - 12.0

def K_clust_on_line(K, per_clust, ep=0.1):
    #Build the boundries between clusters
    gap = 10.0 / float(K - 1)
    boundaries = np.zeros(K+1)
    offset = -5.0 - gap/2.0
    for k in xrange(K+1):
        boundaries[k] = offset + k*gap

    #Build the clusters
    x = gap / (per_clust +ep)
    y = x + ep
    n = K * per_clust
    X = np.zeros(n)
    for k in xrange(K):
        for j in xrange(per_clust):
            index = k*per_clust + j
            X[index] = boundaries[k] + 0.5*y + j*x
    return X.reshape(n, 1)


def _compute_global_opt(clusters, centres, d, n):
    global_opt = [np.sum((clust - c) ** 2.0) for clust, c in zip(clusters, centres)]
    global_opt = sum(global_opt) / float(d * n)
    return global_opt


def is_too_close(new_centre, centres, dist):
    for c in centres:
        if np.linalg.norm(new_centre - c) < dist:
            return True
    return False



# clusters, centres = k_gaussian_clusters(4)
# draw(clusters, centres)




