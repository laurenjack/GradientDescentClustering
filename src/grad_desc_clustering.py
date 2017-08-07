import numpy as np


class GDC:


    def train(self, X, K, lr, epochs, X_bar=None):
        n, d = X.shape
        #Initialize parameters
        if X_bar is None:
            X_bar = np.random.uniform(low= -5.0, high=5.0, size=(K, d))
        W = np.random.randn(n, K)
        all_prev = []
        #Train
        all_dW = np.zeros(epochs)
        all_dX_bar = np.zeros(epochs)
        for e in xrange(epochs):
            all_prev.append(X_bar)
            dW, dX_bar = self.compute_grads(X, X_bar, W)
            W -= lr*dW
            X_bar = X_bar - lr*dX_bar
            all_dW[e] = np.mean(abs(dW))
            all_dX_bar[e] = np.mean(abs(dX_bar))
        return W, X_bar, all_prev, (all_dW, all_dX_bar)

    def compute_grads(self, X, X_bar, W):
        p = self._compute_p(W)
        x_diff = self._compute_diff(X, X_bar)
        #Gradient w.r.t the weights
        dist_sq = x_diff ** 2.0
        #Scale by average p *(1 - p)
        #p_prod = p * (1 - p)
        n, K = p.shape
        _, d = X_bar.shape
        p_broadcast = p.reshape(n, K, 1)
        sum_dist_sq = np.sum(dist_sq, axis=2)
        C = p * sum_dist_sq
        #pp_mean = np.mean(p_prod) #+ 0.0000001
        dW = p * (sum_dist_sq - np.sum(C, axis=1).reshape((n, 1)))
        #dW = p_prod * np.sum(dist_sq, axis=2) #/pp_mean
        dW = dW/(np.mean(abs(dW)))
        #Gradient w.r.t the centres
        dX_bar = np.sum(p_broadcast * -x_diff, axis = 0) * 2.0
        over_n = 1.0/float(n)
        over_d = 1.0/float(d)
        c = 1.0 #over_d
        return c * dW, c * dX_bar #, p_prod#/pp_mean

    def cost(self, X, W, X_bar):
        n, K = W.shape
        #Distances
        x_diff = self._compute_diff(X, X_bar)
        dist_sq = np.sum(x_diff ** 2.0, axis=2)
        #ps
        p = self._compute_p(W)
        C = np.sum(p * dist_sq)/float(n)
        return K ** 2.0 * C

    def train_and_cost(self, tp, K):
        """Train GDC adn report the cost"""
        W, X_bar, _ = self.train(tp.X, K, tp.lr, tp.epochs)
        return self.cost(tp.X, W, X_bar)


    def _compute_p(self, W):
        n, _ = W.shape
        exp_W = np.exp(W)
        p = exp_W / np.sum(exp_W, axis=1).reshape(n, 1)
        return p

    def _compute_diff(self, X, X_bar):
        n, d = X.shape
        K, _ = X_bar.shape
        X_3 = X.reshape(n, 1, d)
        X_bar_3 = X_bar.reshape(1, K, d)
        return X_3 - X_bar_3


class Gradients:
    """Class responsible for holding tracked gradients
    over time"""

    def __init__(self, last_p_prod, last_dW, last_dX_bar):
        self.last_p_prod = last_p_prod
        self.last_dW = last_dW
        self.last_dX_bar = last_dX_bar


