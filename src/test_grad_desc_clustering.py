import numpy as np
from unittest import TestCase
from grad_desc_clustering import *

class TestGdc(TestCase):

    def test_compute_gradients(self):
        #Test case has 2 dimensions, 3 centres and 4 observations
        W = np.array([[2.0, 0.5, 0.1], [1.2, 0.4, 2.0], [0.1, 1.0, 0.5], [0.3, 2.0, 0.6]])
        X = np.array([[1.5, 2.5], [3.0, 0.0], [-1.0, 1.0], [-0.5, 2.0]])
        X_bar = np.array([[-0.5, 0.5], [-2.0, -1.0], [1.1, 1.1]])

        #Will compute expected values for an individual centre and two individual weights
        exp_dW_01 = 0.16254852375 * (1.0 - 0.16254852375)  * 24.5 / 4.0 # p*(1-p) * distance^2 of ob0 and centre1 over n
        exp_dW_12 = 0.60561080896 * (1.0 - 0.60561080896) * 4.82 / 4.0 # p*(1-p) * distance^2 of ob1 and centre2 over n
        exp_centre0 = (0.72849194231 * np.array([-2.0, -2.0]) + 0.27211847744 * np.array([-3.5, 0.5])
                       + 0.20196194686 * np.array([0.5, -0.5]) + 0.12781502692 * np.array([0.0, -1.5]))/2.0

        #Compute actual gradients
        gdc = GDC()
        dW, dX_bar = gdc.compute_grads(X, X_bar, W)

        #Assertions
        self.assertAlmostEqual(exp_dW_01, dW[0, 1])
        self.assertAlmostEqual(exp_dW_12, dW[1, 2])
        self.assertTrue(np.allclose(exp_centre0, dX_bar[0]))