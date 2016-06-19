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

from filterpy.stats import norm_cdf, multivariate_gaussian
import numpy as np


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
