# -*- coding: utf-8 -*-
"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import division
from filterpy.common import mahalanobis
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

import numpy as np
from numpy.linalg import inv

ITERS = 10000

def test():
    global a, b, S
    # int test
    a, b, S = 3, 1, 2
    assert abs(mahalanobis(a, b, S) - scipy_mahalanobis(a, b, 1/S)) < 1.e-12


    # int list
    assert abs(mahalanobis([a], [b], [S]) - scipy_mahalanobis(a, b, 1/S)) < 1.e-12
    assert abs(mahalanobis([a], b, S) - scipy_mahalanobis(a, b, 1/S)) < 1.e-12


    # float
    a, b, S = 3.123, 3.235235, .01234
    assert abs(mahalanobis(a, b, S) - scipy_mahalanobis(a, b, 1/S)) < 1.e-12
    assert abs(mahalanobis([a], [b], [S]) - scipy_mahalanobis(a, b, 1/S)) < 1.e-12
    assert abs(mahalanobis([a], b, S) - scipy_mahalanobis(a, b, 1/S)) < 1.e-12

    #float array
    assert abs(mahalanobis(np.array([a]), b, S) - scipy_mahalanobis(a, b, 1/S)) < 1.e-12

    #1d array
    a = np.array([1., 2.])
    b = np.array([1.4, 1.2])
    S = np.array([[1., 2.], [2., 4.001]])

    assert abs(mahalanobis(a, b, S) - scipy_mahalanobis(a, b, inv(S))) < 1.e-12

    #2d array
    a = np.array([[1., 2.]])
    b = np.array([[1.4, 1.2]])
    S = np.array([[1., 2.], [2., 4.001]])

    assert abs(mahalanobis(a, b, S) - scipy_mahalanobis(a, b, inv(S))) < 1.e-12
    assert abs(mahalanobis(a.T, b, S) - scipy_mahalanobis(a, b, inv(S))) < 1.e-12
    assert abs(mahalanobis(a, b.T, S) - scipy_mahalanobis(a, b, inv(S))) < 1.e-12
    assert abs(mahalanobis(a.T, b.T, S) - scipy_mahalanobis(a, b, inv(S))) < 1.e-12

    try:
        # mismatched shapes
        mahalanobis([1], b, S)
        assert "didn't catch vectors of different lengths"
    except ValueError:
        pass
    except:
        assert "raised exception other than ValueError"

    # okay, now check for numerical accuracy
    for _ in range(ITERS):
        N = np.random.randint(1, 20)
        a = np.random.randn(N)
        b = np.random.randn(N)
        S = np.random.randn(N, N)
        S = np.dot(S, S.T) #ensure positive semi-definite
        assert abs(mahalanobis(a, b, S) - scipy_mahalanobis(a, b, inv(S))) < 1.e-12







if __name__ == "__main__":

    ITERS = 1000000
    test()


