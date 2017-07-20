class ConvexOptimizer:

    def __init__(self, algo):
        self.algo = algo

    def bin_search(self, tp, low, high):
        C_low = self.algo.train_and_cost(tp, low)
        C_high = self.algo.train_and_cost(tp, high)
        ps = self._compute_point_set(tp, (low, C_low), (high, C_high))
        return self._bin_search(tp, ps)

    def _bin_search(self, tp, ps):
        left, middle, right = self._compute_lmr(tp, ps)
        # Terminating cases
        if left.mid_is_min() and left.is_final:
            return left.k_mid[1]
        if right.mid_is_min() and right.is_final:
            return right.k_mid[1]
        if middle.mid_is_min() and middle.is_final:
            return middle.k_mid[1]
        # Recursive cases
        if left.mid_is_min():
            return self._bin_search(tp, left)
        if right.mid_is_min():
            return self._bin_search(tp, right)
        return self._bin_search(tp, middle)

    def _compute_lmr(self, tp, ps):
        left = self._compute_point_set(tp, ps.k_low, ps.k_mid)
        right = self._compute_point_set(tp, ps.k_mid, ps.k_high)
        middle_is_final = ps.k_mid[0] <= left.k_mid[0] + 1
        middle = PointSet(left.k_mid, ps.k_mid, right.k_mid, middle_is_final)
        return left, middle, right

    def _compute_point_set(self, tp, k_low, k_high):
        low, C_low = k_low
        high, C_high = k_high
        mid = low + (high - low + 1) // 2
        is_final = mid <= low + 1
        C = self.algo.train_and_cost(tp, mid)
        return PointSet(k_low, (mid, C), k_high, is_final)



class TrainingParams:

    def __init__(self, X, lr, epochs):
        self.X = X
        self.lr = lr
        self.epochs = epochs

class PointSet:

    def __init__(self, k_low, k_mid, k_high, is_final):
        self.k_low = k_low
        self.k_mid = k_mid
        self.k_high = k_high
        self.is_final = is_final

    def mid_is_min(self):
        return self.k_mid[1] < self.k_low[1] and self.k_mid[1] < self.k_high[1]