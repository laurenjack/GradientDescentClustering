import numpy as np
import math

class GDC:


    def train_groups(self, X, K, lr, epochs, X_bar):
        all_prev = []
        #Create groups
        _, d = X_bar.shape
        half_way = K/2
        # firsts = X_bar[:half_way]
        # seconds = X_bar[half_way:]
        first_inds = np.arange(half_way/2)
        second_inds = np.arange(half_way/2)

        for e in xrange(epochs):
            # first0 = X_bar[0].reshape(1, d)
            # first1 = X_bar[1].reshape(1, d)
            # second0 = X_bar[half_way].reshape(1, d)
            second1 = X_bar[half_way+1].reshape(1, d)
            all_prev.append(np.copy(X_bar))
            #all_prev.append(np.concatenate((first0, first1, second0, second1), axis=0))
            np.random.shuffle(first_inds)
            np.random.shuffle(second_inds)
            for i in xrange(0, half_way/2, 1):
                f = first_inds[i]*2
                s = second_inds[i]*2
                first0 = X_bar[f].reshape(1, d)
                first1 = X_bar[f+1].reshape(1, d)
                second0 = X_bar[half_way+s].reshape(1, d)
                second1 = X_bar[half_way+s+1].reshape(1, d)
                training_x_bar = np.concatenate((first0, first1, second0, second1), axis=0)
                #Actually train it now
                dX_bar, p = self.compute_grads_cost_as_weight(X, training_x_bar)
                X_bar[f] -= lr * dX_bar[0]
                X_bar[f+1] -= lr * dX_bar[1]
                X_bar[half_way+s] -= lr * dX_bar[2]
                X_bar[half_way+s+1] -= lr * dX_bar[3]

        return p, X_bar, all_prev, None



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
        for e in xrange(epochs):
            # if e == 100:
            #     lr *= 0.001
            all_prev.append(X_bar)
            if L is None:
                dX_bar, p = self.compute_grads_soft_rbf(X, X_bar)
                dW = 1.0
                #dW, dX_bar, p = self.compute_grads(X, X_bar, W)
            else:
                dW, dX_bar, p = self.compute_grads_redundancy(X, X_bar, W)
                #dX_bar, p = self.compute_grads_cost_as_weight(X, X_bar)
                #dW = p
                #dW, dX_bar, p = self.compute_grads_rbf(X, X_bar, W)
            #dW, dX_bar, p = self.compute_grads_per_clust_cost(X, X_bar, W)
            #W -= lr*dW
            X_bar = X_bar + lr*dX_bar
            mag_dW = abs(dW)
            mag_dX_bar = abs(dX_bar)
            all_dW[e] = np.mean(mag_dW)
            all_dX_bar[e] = np.mean(mag_dX_bar)
            dW_max[e] = np.max(mag_dW)
            dX_bar_max[e] = np.max(mag_dX_bar)
            p_sum_max[e] = np.max(np.sum(p, axis=0))
        return p, X_bar, all_prev, GradStats(all_dW, all_dX_bar, dW_max, dX_bar_max, p_sum_max)

    def train_sgd(self, X, K, lr, epochs, m, X_bar, weights=False):
        n, d = X.shape
        w_lr = 0.01
        z = 0.2#* np.ones((K, d))
        all_prev_z = np.zeros(epochs)
        if weights:
            z *= np.ones((K, d))
        W = np.random.randn(n, K)
        all_prev = []
        # Train
        batch_indicies = np.arange(n)
        p = np.zeros((n, X_bar.shape[0]))
        for e in xrange(epochs):
            if (e+1) % (epochs / 4) == 0:
                lr *= 0.35
            if(e+1) % (3 * epochs / 4) == 0:
                m = n
                w_lr *= 0.1
            all_prev.append(X_bar)
            X_bar = np.copy(X_bar)
            #np.random.shuffle(batch_indicies)
            for k in xrange(0, n, m):
                batch = batch_indicies[k:k + m]
                #dX_bar, p[batch], dC_dz = self.compute_grads_soft_rbf(X[batch], X_bar, z)
                if weights:
                    dX_bar, p[batch], dJ_dz = self.cg_dual_p(X[batch], X_bar, z)
                    z += w_lr * dJ_dz
                else:
                    dX_bar, p[batch], dJ_dz, dp_dz = self.compute_grads_soft_rbf_test(X[batch], X_bar, z)
                    all_prev_z[e] = z
                    #z -= 0.1 * z * np.sign(dJ_dz)
                #dW, dX_bar, p[batch] = self.compute_grads_exp(X[batch], X_bar, W[batch])
                #dW, dX_bar, mm_dW, p[batch] = self.compute_grads(X[batch], X_bar, W[batch], mm_dW)
                # dW, dX_bar, p = self.compute_grads_per_clust_cost(X, X_bar, W)
                #W[batch] -= lr * dW
                X_bar = X_bar + lr * dX_bar
                # z += 0.01 * dJ_dz
        print "w: "+str(z)
        return p, X_bar, all_prev, all_prev_z

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

    def compute_grads_cost_as_weight(self, X, X_bar):
        x_diff = self._compute_diff(X, X_bar)
        dist_sq = x_diff ** 2.0
        C_ik = np.sum(dist_sq, axis=2)
        p = self._compute_p(-C_ik)
        n, K = p.shape
        _, d = X_bar.shape
        p_broadcast = p.reshape(n, K, 1)
        dist_weighted_diff = -x_diff * C_ik.reshape(n, K, 1)
        dist_weighted_diff_per_K = np.sum(p_broadcast * dist_weighted_diff, axis=1).reshape(n, 1, d)
        dX_bar = 2.0 * np.sum(p.reshape(n, K, 1) * (-x_diff - (dist_weighted_diff - dist_weighted_diff_per_K)), axis=0)
        dX_bar /= np.sum(dX_bar ** 2.0) ** 0.5 + 0.001
        return dX_bar, p

    def compute_grads_redundancy(self, X, X_bar, W):
        n, _ = X.shape
        p = self._compute_p(W)
        L, K = p.shape
        x_diff = self._compute_diff(X, X_bar)
        dist_sq = x_diff ** 2.0
        sq_sum_nk = np.sum(dist_sq, axis=2)
        closest_clust = np.argmin(sq_sum_nk, axis=1)
        min_sq_sum_n = np.amin(sq_sum_nk, axis=1)
        lowered_cost = np.repeat(min_sq_sum_n.reshape(n, 1), L, axis=1)
        #cost_funct_weighting = np.exp(min_sq_sum_n) / np.sum(np.exp(min_sq_sum_n))
        x_diff_norm =  x_diff/sq_sum_nk.reshape(n, L, 1) ** 0.5
        lowered_x_diff = lowered_cost.reshape(n, L, 1) ** 0.5 * x_diff_norm

        sq_sum_k = np.sum(sq_sum_nk, axis=0)
        x_diff_sum_kd = np.sum(lowered_x_diff , axis=0)

        #dW = p * (sq_sum_k.reshape(1, K) - np.sum(p * sq_sum_k, axis=1).reshape(L, 1))
        dW = p * sq_sum_k - p * np.sum(p * sq_sum_k, axis=1).reshape(L, 1)
        #dW /= np.sum(dW ** 2.0) ** 0.5 + 0.001

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


    def compute_grads_soft_rbf(self, X, X_bar, z):
        n, d = X.shape
        K, _ = X_bar.shape
        x_diff = self._compute_diff(X, X_bar)
        dist_sq = x_diff ** 2.0
        sq_sum_nk = np.sum(dist_sq, axis=2)
        #rbf = self.rbf(0.5 * sq_sum_nk)
        rbf = self.rbf(z ** 2.0 / float(d) * sq_sum_nk)
        #scale = self._compute_p(-10*sq_sum_nk)
        p = self._compute_p(200.0 * rbf)
        dX_bar_n = z ** 2.0 * x_diff * (p * rbf).reshape(n, K, 1)
        mags = np.sum(dX_bar_n ** 2.0, axis=2) ** 0.5
        avg_mag = np.mean(mags, axis=0)
        dX_bar = 2 / float(n) * np.sum(dX_bar_n, axis=0) / (avg_mag.reshape(K, 1) + 10 ** (-120))
        #dX_bar = 2 * np.sum(x_diff * (scale * rbf).reshape(n, K, 1), axis=0)
        #dX_bar/= np.sum(dX_bar ** 2.0) ** 0.5 + 0.001
        #Compute the z grad
        dp_dz = 2 * z * p * rbf * (-sq_sum_nk + np.sum(p * sq_sum_nk, axis=1).reshape(n, 1))
        dC_dz = np.sum(dp_dz * sq_sum_nk)
        return dX_bar, p, dC_dz, dp_dz

    def compute_grads_soft_rbf_test(self, X, X_bar, z):
        n, d = X.shape
        K, _ = X_bar.shape
        x_diff = self._compute_diff(X, X_bar)
        dist_sq = x_diff ** 2.0
        sq_sum_nk = np.sum(dist_sq, axis=2)
        #rbf = self.rbf(0.5 * sq_sum_nk)
        rbf = self.rbf(z ** 2.0 / float(d) * sq_sum_nk)
        #scale = self._compute_p(-10*sq_sum_nk)
        p = self._compute_p(30.0 * rbf)
        dX_bar_n = z ** 2.0 * x_diff * (p * rbf).reshape(n, K, 1)
        mags = np.sum(dX_bar_n ** 2.0, axis=2) ** 0.5
        avg_mag = np.mean(mags, axis=0)
        dX_bar = 2 / float(n) * np.sum(dX_bar_n, axis=0) / (avg_mag.reshape(K, 1) + 10 ** (-120))
        to_7 = 2 / float(n) * np.sum(dX_bar_n[:7], axis=0) / avg_mag.reshape(K, 1)
        to_12 = 2 / float(n) * np.sum(dX_bar_n[7:], axis=0)/ avg_mag.reshape(K, 1)
        #dX_bar = 2 * np.sum(x_diff * (scale * rbf).reshape(n, K, 1), axis=0)
        #dX_bar/= np.sum(dX_bar ** 2.0) ** 0.5 + 0.001
        #Compute the z grad
        dX_bar_n_scaled =   2 / float(n) * dX_bar_n[:,:,0] / avg_mag
        dp_dz = 2 * z * p * (-rbf * sq_sum_nk + np.sum(p * rbf * sq_sum_nk, axis=1).reshape(n, 1))
        dC_dz = np.sum(dp_dz * sq_sum_nk)
        return dX_bar, p, dC_dz, dp_dz

    def cg_soft_rbf_weights(self, X, X_bar, z):
        n, d = X.shape
        K, _ = X_bar.shape
        x_diff = self._compute_diff(X, X_bar)
        dist_sq = x_diff ** 2.0
        w_dist_sq = z ** 2.0 * dist_sq
        sq_sum_nk = np.sum(dist_sq, axis=2)
        w_sq_sum_nk = np.sum(w_dist_sq, axis=2)
        #rbf = self.rbf(0.5 * sq_sum_nk)
        rbf = self.rbf(w_sq_sum_nk / float(d))
        #scale = self._compute_p(-10*sq_sum_nk)
        p = self._compute_p(200 * rbf)
        dX_bar_n = z ** 2.0 * x_diff * (p * rbf).reshape(n, K, 1)
        mags = np.sum(dX_bar_n ** 2.0, axis=2) ** 0.5
        avg_mag = np.mean(mags, axis=0)
        dX_bar = 2 * np.sum(dX_bar_n, axis=0) / (avg_mag.reshape(K, 1) + 10 ** (-120))
        #dX_bar = 2 * np.sum(x_diff * (scale * rbf).reshape(n, K, 1), axis=0)
        #dX_bar/= np.sum(dX_bar ** 2.0) ** 0.5 + 0.001
        #Compute the z grad
        p_shaped = p.reshape(n, K, 1)
        w_shaped = z.reshape(1, 1, d)
        dp_dz = 2 * w_shaped * (p * rbf).reshape(n, K, 1) * (-dist_sq + np.sum(p_shaped * dist_sq, axis=1).reshape(n, 1, d))
        dC_dz = np.sum(dp_dz * w_sq_sum_nk.reshape(n, K, 1), axis=(0, 1))
        dC_dz /= np.sum(dC_dz ** 2.0) ** 0.5 + 10 ** (-120)
        return dX_bar, p, dC_dz

    def cg_dual_p(self, X, X_bar, w):
        n, d = X.shape
        K, _ = X_bar.shape
        x_diff = self._compute_diff(X, X_bar)
        dist_sq = x_diff ** 2.0
        sq_sum_nk = np.sum(dist_sq, axis=2)
        w_sq_sum_nk = np.sum(w ** 2.0 * dist_sq, axis=2)
        rbf = self.rbf(w_sq_sum_nk / float(d))
        p = self._compute_p(200.0 * rbf)
        closest_to = np.argmin(w_sq_sum_nk, axis=1)
        p2 = np.zeros((n, K))
        i_indices = np.arange(n)
        #p2[i_indices, closest_to] = 1.0
        p2 = self._compute_p(- w_sq_sum_nk)
        w_shaped = w.reshape(1, K, d)
        dX_bar_n = 4 * w_shaped ** 2.0 * x_diff * (rbf * p).reshape(n, K, 1) # * (p - np.sum(p ** 2.0, axis=1).reshape(n, 1))
        mags = np.sum(dX_bar_n ** 2.0, axis=2) ** 0.5
        avg_mag = np.mean(mags, axis=0)
        dX_bar = np.sum(dX_bar_n, axis=0) / (avg_mag.reshape(K, 1) + 10 ** (-120))
        current_vars = np.sum(p2.reshape(n, K, 1) * dist_sq, axis=0) / float(n)
        dJ_dw = current_vars ** 0.5 - w
        dJ_dw /= (np.sum(dJ_dw ** 2.0, axis=1) ** 0.5).reshape(K, 1)
        # dJ_dw_n = - 4 * w_shaped * dist_sq * (rbf * p * (p - np.sum(p ** 2.0, axis=1).reshape(n, 1))).reshape(n, K, 1)
        # mags = np.sum(dJ_dw_n ** 2.0, axis=2) ** 0.5
        # avg_mag = np.mean(mags, axis=0)
        # dJ_dw = np.sum(dJ_dw_n, axis=0) / (avg_mag.reshape(K, 1) + 10 ** (-120))
        return dX_bar, p, dJ_dw

    def cg_fit_gauss(self, x_diff, w):
        dist_sq = x_diff ** 2.0
        _sq_sum_nk = np.sum(w ** 2.0 * dist_sq, axis=2)
        rbf = self.rbf(w_sq_sum_nk / float(d))



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
        """Train GDC and report the cost"""
        W, X_bar, _ = self.train(tp.X, K, tp.lr, tp.epochs)
        return self.cost(tp.X, W, X_bar)


    def _compute_p(self, W):
        n, _ = W.shape
        exp_W = np.exp(W)
        p = exp_W / (np.sum(exp_W, axis=1).reshape(n, 1) + 10 ** (-120))
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





