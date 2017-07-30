import numpy as np


class KMeans:

    def train(self, X, K, epochs, X_bar=None):
        n, d = X.shape
        # Initialize parameters
        if X_bar is None:
            X_bar = np.random.uniform(low=-5.0, high=5.0, size=(K, d))
        all_prev = []
        # Train
        for e in xrange(epochs):
            all_prev.append(X_bar)
            x_diff = _compute_diff(X, X_bar)
            dist_sq = x_diff ** 2.0
            dist_from_each = np.sum(dist_sq, axis=2)
            closest_to = np.argmin(dist_from_each, axis=1)
            total = np.zeros((K, d))
            sizes = np.zeros(K)
            for i in xrange(n):
                k = closest_to[i]
                total[k] += X[i]
                sizes[k] += 1
            sizes = sizes.reshape(K, 1)
            X_bar = total / np.where(sizes > 0, sizes, np.ones((K, 1)))

        return X_bar, all_prev





def _compute_diff(X, X_bar):
    n, d = X.shape
    K, _ = X_bar.shape
    X_3 = X.reshape(n, 1, d)
    X_bar_3 = X_bar.reshape(1, K, d)
    return X_3 - X_bar_3

