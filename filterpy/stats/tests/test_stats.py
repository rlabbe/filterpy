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
import scipy
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

    try:
        multivariate_gaussian(1, 1, -1)
    except:
        pass
    else:
        assert False, "negative variances are meaningless"

    # test that we get the same results as scipy.stats.multivariate_normal
    xs = np.random.randn(1000)
    mean = np.random.randn(1000)
    var = np.random.random(1000) * 5

    for x, m, v in zip(xs, mean, var):
        assert abs(multivariate_gaussian(x, m, v) - scipy.stats.multivariate_normal(m, v).pdf(x)) < 1.e-12


def _is_inside_ellipse(x, y, ex, ey, orientation, width, height):

    co = np.cos(orientation)
    so = np.sin(orientation)

    xx = x*co + y*so
    yy = y*co - x*so

    return (xx / width)**2 + (yy / height)**2 <= 1.


def do_plot_test():
    import matplotlib.pyplot as plt
    from numpy.random import multivariate_normal as mnormal
    from filterpy.stats import covariance_ellipse, plot_covariance

    p = np.array([[32, 15], [15., 40.]])

    x, y = mnormal(mean=(0, 0), cov=p, size=5000).T
    sd = 2
    a, w, h = covariance_ellipse(p, sd)
    print(np.degrees(a), w, h)

    count = 0
    color = []
    for i in range(len(x)):
        if _is_inside_ellipse(x[i], y[i], 0, 0, a, w, h):
            color.append('b')
            count += 1
        else:
            color.append('r')
    plt.scatter(x, y, alpha=0.2, c=color)
    plt.axis('equal')

    plot_covariance(mean=(0., 0.),
                    cov=p,
                    std=[1,2,3],
                    alpha=0.3,
                    facecolor='none')

    print(count / len(x))

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


def covariance_3d_plot_test():
    import matplotlib.pyplot as plt
    from filterpy.stats import plot_3d_covariance

    mu = [13456.3,2320,672.5]

    C = np.array([[1.0, .03, .2],
                  [.03,  4.0, .0],
                  [.2,  .0, 16.1]])

    sample = np.random.multivariate_normal(mu, C, size=1000)

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=sample[:, 0], ys=sample[:, 1], zs=sample[:, 2], s=1)
    plot_3d_covariance(mu, C, alpha=.4, std=3, limit_xyz=True, ax=ax)

if __name__ == "__main__":
    covariance_3d_plot_test()
    plt.figure()
    do_plot_test()

