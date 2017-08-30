import numpy as np
import math

class GDC:


    def train(self, X, K, lr, epochs, X_bar=None, L=None):
        n, d = X.shape
        #Initialize parameters
        if X_bar is None:
            X_bar = np.random.uniform(low= -5.0, high=5.0, size=(K, d))
        if L is None:
            W = np.random.randn(n, K)
        else:
            W = np.random.randn(L, K)
        all_prev = []
        #Train
        all_dW = np.zeros(epochs)
        all_dX_bar = np.zeros(epochs)
        dW_max = np.zeros(epochs)
        dX_bar_max = np.zeros(epochs)
        p_sum_max = np.zeros(epochs)
        mm_dW = 1.0
        for e in xrange(epochs):
            # if e == 100:
            #     lr *= 0.001
            all_prev.append(X_bar)
            # if e % 20 == 0:
            #     is_exp = not is_exp
            if L is None:
                dW, dX_bar, p = self.compute_grads(X, X_bar, W)
            else:
                dW, dX_bar, p = self.compute_grads_rbf(X, X_bar, W)
            #dW, dX_bar, p = self.compute_grads_per_clust_cost(X, X_bar, W)
            W -= lr*dW
            X_bar = X_bar - lr*dX_bar
            mag_dW = abs(dW)
            mag_dX_bar = abs(dX_bar)
            all_dW[e] = np.mean(mag_dW)
            all_dX_bar[e] = np.mean(mag_dX_bar)
            dW_max[e] = np.max(mag_dW)
            dX_bar_max[e] = np.max(mag_dX_bar)
            p_sum_max[e] = np.max(np.sum(p, axis=0))
        return p, X_bar, all_prev, GradStats(all_dW, all_dX_bar, dW_max, dX_bar_max, p_sum_max)

    def train_sgd(self, X, K, lr, epochs, m, X_bar):
        n, d = X.shape
        W = np.random.randn(n, K)
        all_prev = []
        # Train
        mm_dW = 1.0
        batch_indicies = np.arange(n)
        p = np.zeros((n, K))
        for e in xrange(epochs):
            if (e+1) % 60 == 0:
                lr *= 0.1
            all_prev.append(X_bar)
            X_bar = np.copy(X_bar)
            np.random.shuffle(batch_indicies)
            for k in xrange(0, n, m):
                batch = batch_indicies[k:k + m]
                dW, dX_bar, p[batch] = self.compute_grads_exp(X[batch], X_bar, W[batch])
                #dW, dX_bar, mm_dW, p[batch] = self.compute_grads(X[batch], X_bar, W[batch], mm_dW)
                # dW, dX_bar, p = self.compute_grads_per_clust_cost(X, X_bar, W)
                W[batch] -= lr * dW
                X_bar = X_bar - lr * dX_bar
        return p, X_bar, all_prev

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
        #mm_dW += 0.1 * (np.mean(abs(dW)) - mm_dW)
        #dW = dW / (mm_dW + 1.0)
        #dW = dW/(np.mean(abs(dW)))
        #Gradient w.r.t the centres
        dX_bar = np.sum(p_broadcast * -x_diff, axis = 0) * 2.0
        #dX_bar /= mm_dW + 1.0
        #dX_bar = dX_bar/(np.mean(abs(dX_bar)) + 0.1)
        mag_sf = (np.sum(dW ** 2.0) + np.sum(dX_bar ** 2.0)) ** 0.5
        dW/= np.sum(dW ** 2.0) ** 0.5 + 0.001
        #return dW / (mag_sf), dX_bar / (mag_sf), mm_dW, p
        return dW, dX_bar, p

    def compute_grads_redundancy(self, X, X_bar, W):
        p = self._compute_p(W)
        L, K = p.shape
        x_diff = self._compute_diff(X, X_bar)
        dist_sq = x_diff ** 2.0
        sq_sum_nk = np.sum(dist_sq, axis=2)
        sq_sum_k = np.sum(sq_sum_nk, axis=0)
        x_diff_sum_kd = np.sum(x_diff, axis=0)

        #dW = p * (sq_sum_k.reshape(1, K) - np.sum(p * sq_sum_k, axis=1).reshape(L, 1))
        dW = p * sq_sum_k - p * np.sum(p * sq_sum_k, axis=1).reshape(L, 1)
        dW /= np.sum(dW ** 2.0) ** 0.5 + 0.001

        p_sum = np.sum(p, axis=0)
        dX_bar = 2.0 * p_sum.reshape(K, 1) * -x_diff_sum_kd

        return dW, dX_bar, p
        # _, d = X_bar.shape

    def compute_grads_rbf(self, X, X_bar, W):
        p = self._compute_p(W)
        n, d = X.shape
        L, K = p.shape
        x_diff = self._compute_diff(X, X_bar)
        dist_sq = x_diff ** 2.0
        sq_sum_nk = np.sum(dist_sq, axis=2)
        rbf = self.rbf(sq_sum_nk)
        sum_rbf_k = rbf.sum(axis=0)
        d_rbf_d_x_bar_n = (2.0 / math.pi) ** 0.5 * - x_diff *  rbf.reshape(n, K, 1)
        d_rbf_d_x_bar = np.sum(d_rbf_d_x_bar_n, axis=0)
        p_sum = np.sum(p, axis=0)
        dX_bar = p_sum.reshape(K, 1) * d_rbf_d_x_bar

        dW = p * (sum_rbf_k.reshape(1, K) - np.sum(p * sum_rbf_k, axis=1).reshape(L, 1))


        #dX_bar = 2.0 * p_sum.reshape(K, 1) * -x_diff_sum_kd

        return dW, dX_bar, p


    def compute_grads_exp(self, X, X_bar, W):
        p = self._compute_p(W)
        x_diff = self._compute_diff(X, X_bar)
        #Gradient w.r.t the weights
        dist_sq = x_diff ** 2.0
        n, K = p.shape
        _, d = X_bar.shape
        p_broadcast = p.reshape(n, K, 1)
        sum_dist_sq = np.sum(dist_sq, axis=2)
        Cik = p * sum_dist_sq
        Ci = np.sum(Cik, axis=1).reshape((n, 1))
        exp_Ci = np.exp(Ci)
        dW = p * (sum_dist_sq - Ci) * exp_Ci
        #Gradient w.r.t the centres
        dX_bar = np.sum(p_broadcast * -x_diff * exp_Ci.reshape((n, 1, 1)), axis = 0) * 2.0
        mag_sf = (np.sum(dW ** 2.0) + np.sum(dX_bar ** 2.0)) ** 0.5

        return dW / (mag_sf + 0.1), dX_bar / (mag_sf + 0.1), p

    def compute_grads_per_clust_cost(self, X, X_bar, W):
        p = self._compute_p(W)
        x_diff = self._compute_diff(X, X_bar)
        #Gradient w.r.t the weights
        dist_sq = x_diff ** 2.0
        n, K = p.shape
        _, d = X_bar.shape
        p_broadcast = p.reshape(n, K, 1)
        sum_dist_sq = np.sum(dist_sq, axis=2)
        p_sum = np.sum(p, axis=0)
        p_sum_broadcast = p_sum.reshape(K, 1)

        Cik_p_sum = sum_dist_sq * p_sum
        Cik_E = p * sum_dist_sq
        Ck = np.sum(Cik_E, axis=0)
        Cik_pc = (Cik_p_sum - Ck)/p_sum ** 2.0
        Cik_pc_E = p * Cik_pc
        Ci_pc = np.sum(Cik_pc_E, axis=1)
        dW = p*(Cik_pc - Ci_pc.reshape(n, 1))

        dX_bar = np.sum(p_broadcast * -x_diff, axis=0) * 2.0 / p_sum_broadcast
        mag_sf = (np.sum(dW ** 2.0) + np.sum(dX_bar ** 2.0)) ** 0.5

        return dW/mag_sf, dX_bar/mag_sf, p

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

    def rbf(self, sq_sum_nk):
        return np.exp(-sq_sum_nk)


class GradStats:

    def __init__(self, avg_dW, avg_dX_bar, max_dW, max_dX_bar, p_sum_max):
        self.avg_dW = avg_dW
        self.avg_dX_bar = avg_dX_bar
        self.max_dW = max_dW
        self.max_dX_bar = max_dX_bar
        self.p_sum_max = p_sum_max

    def as_np(self):
        epochs = self.avg_dW.shape[0]
        grad_aggs = [self.avg_dW, self.avg_dX_bar, self.max_dW, self.max_dX_bar, self.p_sum_max]
        grad_aggs = [agg.reshape(1, epochs) for agg in grad_aggs]
        return np.concatenate(grad_aggs, axis=0)





