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

from filterpy.discrete_bayes import predict, update, normalize
from numpy.random import randn, randint
import numpy as np


def _predict(distribution, offset, kernel):
    """ explicit convolution with wraparound"""

    N = len(distribution)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i in range(N):
        for k in range (kN):
            index = (i + (width-k) - offset) % N
            prior[i] += distribution[index] * kernel[k]
    return prior

    
def test_predictions():
    s = 0.

    for k in range(3, 22, 2):  # different kernel sizes
        for _ in range(1000):
            a = randn(100)
            kernel = normalize(randn(k))
            move = randint(1, 200)
            s += sum(predict(a, move, kernel) - _predict(a, move, kernel))

        assert s < 1.e-8, "sum of difference = {}".format(s)
