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
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

from filterpy.common import mahalanobis
from filterpy.common import kinematic_kf

import numpy as np
from numpy.linalg import inv

ITERS = 10000

def test_mahalanobis():
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



def test_kinematic_filter():
    global kf

    # make sure the default matrices are shaped correctly
    for dim_x in range(1,4):
        for order in range (0, 3):
            kf = kinematic_kf(dim=dim_x, order=order)
            kf.predict()
            kf.update(np.zeros((dim_x, 1)))


    # H is tricky, make sure it is shaped and assigned correctly
    kf = kinematic_kf(dim=2, order=2)
    assert kf.x.shape == (6, 1)
    assert kf.F.shape == (6, 6)
    assert kf.P.shape == (6, 6)
    assert kf.Q.shape == (6, 6)
    assert kf.R.shape == (2, 2)
    assert kf.H.shape == (2, 6)

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]], dtype=float)
    assert np.array_equal(H, kf.H)

    kf = kinematic_kf(dim=3, order=2, order_by_dim=False)
    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=float)
    assert np.array_equal(H, kf.H)


if __name__ == "__main__":

    ITERS = 1000000
    #test_mahalanobis()

    test_kinematic_filter()
