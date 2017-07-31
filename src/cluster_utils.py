import numpy as np

def cost_of_centres(clusters, centres):
    """Find the cost of K clusters from K centres.

    :param clusters - must be equal length to centres,
    the points in the kth cluster must have the kth centre

    :param centres - must be equal length to clusters,
    the kth cluster must have the kth centre

    :return A scalar, returns the dimension weighted Euclidian
    distance of the given clustering arrangement
    """
    K, d = centres.shape
    d = float(d)
    n = float(sum([len(cluster) for cluster in clusters]))
    return sum([sum([np.sum((x - centres[k]) ** 2.0) for x in cluster]) for k in xrange(K)]) /(n * d)

def find_closest_to(X, centres):
    """Associate each point x with the closest centre.

    :param X - an n x d matrix, of n d-dimensional points

    :param centres - a K x d matrix, of d-dimensional centres

    :return: A list of K clusters, where the points in each cluster
    are closest to the k_th centre
    """
    x_diff = _compute_diff(X, centres)
    dist_sq = x_diff ** 2.0
    n, _ = X.shape
    K, _ = centres.shape
    dist_from_each = np.sum(dist_sq, axis=2)
    closest_to = np.argmin(dist_from_each, axis=1)
    clusters = [[]] * K
    #Assign each point to its cluster
    for i in xrange(n):
        k = closest_to[i]
        clusters[k].append(X[i])
    return clusters


def cost_of_closest_to(X, centres):
    clusters = find_closest_to(X, centres)
    return cost_of_centres(clusters, centres)

def _compute_diff(X, X_bar):
    n, d = X.shape
    K, _ = X_bar.shape
    X_3 = X.reshape(n, 1, d)
    X_bar_3 = X_bar.reshape(1, K, d)
    return X_3 - X_bar_3