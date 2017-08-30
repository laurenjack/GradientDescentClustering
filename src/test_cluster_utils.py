import numpy as np
from unittest import TestCase
from cluster_utils import *



X = np.array([[-1.0, 1.0], [2.0, 2.0], [0.5, 0.5], [-0.5, 2.0], [1.0, 0.5], [1.5, 0], [-1.0, 3.0], [2.0, -1.0]])
centres = np.array([[1.0, 0.0], [-1.0, -1.0], [-1.0, 2.0], [2.5, 2.5]])
clusters = [[np.array([0.5, 0.5]), np.array([1.0, 0.5]), np.array([1.5, 0.0]), np.array([2.0, -1.0])],
                        [],
                        [np.array([-1.0, 1.0]), np.array([-0.5, 2.0]), np.array([-1.0, 3.0])],
                        [np.array([2.0, 2.0])]]

class ClusterUtilsSpec(TestCase):



    def test_find_closest_to(self):
        act_clusters = find_closest_to(X, centres)



        self.assertEqual(len(clusters), len(act_clusters))
        for exp_clust, act_clust in zip(clusters, act_clusters):
            self.assertEqual(len(exp_clust), len(act_clust))
            for exp_point, act_point in zip(exp_clust, act_clust):
                self.assertTrue(np.allclose(exp_point, act_point))


    def test_cost_of_centres(self):
        act_cost = cost_of_centres(clusters, centres)

        self.assertAlmostEqual(0.359375, act_cost)
