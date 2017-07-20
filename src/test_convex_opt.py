import numpy as np
from unittest import TestCase
from convex_opt import *

class ConvexOptimzerSpec(TestCase):

    def test_bin_search_immediate_mid(self):
        opt = self._create_opt([15.0, 14.5, 15.0], 5)
        K, cost = opt.bin_search(None, 5, 7)
        self.assertEqual(6, K)
        self.assertEqual(14.5, cost)

    def test_bin_search_just_left(self):
        opt = self._create_opt([1.0, 0.01, 0.5, 0.6], 10)
        K, cost = opt.bin_search(None, 10, 13)
        self.assertEqual(11, K)
        self.assertEqual(0.01, cost)

    def test_bin_search_just_right(self):
        opt = self._create_opt([1.0, 0.5, 0.05, 0.6], 1)
        K, cost = opt.bin_search(None, 1, 4)
        self.assertEqual(3, K)
        self.assertEqual(0.05, cost)

    def test_bin_search_mid_5(self):
        opt = self._create_opt([15.0, 14.5, 5.0, 10, 15.0], 4)
        K, cost = opt.bin_search(None, 4, 8)
        self.assertEqual(6, K)
        self.assertEqual(5.0, cost)

    def test_bin_search_left_5(self):
        opt = self._create_opt([15.0, 3.0, 14.5, 14.75, 15.0], 8)
        K, cost = opt.bin_search(None, 8, 12)
        self.assertEqual(9, K)
        self.assertEqual(3.0, cost)

    def test_bin_search_right_5(self):
        opt = self._create_opt([16.0, 15.0, 14.5, 0.9, 15.0], 2)
        K, cost = opt.bin_search(None, 2, 6)
        self.assertEqual(5, K)
        self.assertEqual(0.9, cost)

    def _create_opt(self, costs, low):
        return ConvexOptimizer(DummyConvex(costs, low))

class DummyConvex:
    """A mock algorithm with a convex cost function which is just specified by a list
    of costs for each K"""

    def __init__(self, costs, low):
        self.costs = costs
        self.low = low

    def train_and_cost(self, tp, K):
        return self.costs[K-self.low]