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

from math import exp
import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from filterpy.stats import norm_cdf, multivariate_gaussian, logpdf, mahalanobis


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


def test_multivariate_gaussian():

    # test that we treat lists and arrays the same
    mean= (0, 0)
    cov=[[1, .5], [.5, 1]]
    a = [[multivariate_gaussian((i, j), mean, cov)
          for i in (-1, 0, 1)]
          for j in (-1, 0, 1)]

    b = [[multivariate_gaussian((i, j), mean, np.asarray(cov))
          for i in (-1, 0, 1)]
          for j in (-1, 0, 1)]

    assert np.allclose(a, b)

    a = [[multivariate_gaussian((i, j), np.asarray(mean), cov)
          for i in (-1, 0, 1)]
          for j in (-1, 0, 1)]
    assert np.allclose(a, b)



def test_norm_cdf():
    # test using the 68-95-99.7 rule

    mu = 5
    std = 3
    var = std*std

    std_1 = (norm_cdf((mu-std, mu+std), mu, var))
    assert abs(std_1 - .6827) < .0001

    std_1 = (norm_cdf((mu+std, mu-std), mu, std=std))
    assert abs(std_1 - .6827) < .0001

    std_1half = (norm_cdf((mu+std, mu), mu, var))
    assert abs(std_1half - .6827/2) < .0001

    std_2 = (norm_cdf((mu-2*std, mu+2*std), mu, var))
    assert abs(std_2 - .9545) < .0001

    std_3 = (norm_cdf((mu-3*std, mu+3*std), mu, var))
    assert abs(std_3 - .9973) < .0001


def test_logpdf():
    assert 3.9 < exp(logpdf(1, 1, .01)) < 4.
    assert 3.9 < exp(logpdf([1], [1], .01)) < 4.
    assert 3.9 < exp(logpdf([[1]], [[1]], .01)) < 4.

    logpdf([1., 2], [1.1, 2], cov=np.array([[1., 2], [2, 5]]), allow_singular=False)
    logpdf([1., 2], [1.1, 2], cov=np.array([[1., 2], [2, 5]]), allow_singular=True)

