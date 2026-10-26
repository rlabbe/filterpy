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

import numpy as np
from filterpy.monte_carlo.resampling import residual_resample


ITERS = 10000


def test_residual_resample_uniform():
    N = 100
    weights = np.full(N, 1./N)
    indexes = residual_resample(weights)

    assert len(indexes) == N
    assert np.all(indexes >= 0) and np.all(indexes < N)
    # uniform weights give each particle floor(N*w) == 1 deterministic copy
    assert np.array_equal(np.sort(indexes), np.arange(N))


def test_residual_resample_skewed():
    # N=3, weights [0.9, 0.09, 0.01]: particles 0 gets 2 deterministic copies,
    # the last slot is drawn from the residual mass [0.7, 0.27, 0.03].
    # With the old bug (residual = w - num_copies) the residuals go negative,
    # the cumulative sum is not sorted, and particles 1 and 2 are never drawn.
    np.random.seed(42)
    weights = [0.9, 0.09, 0.01]
    counts = np.zeros(3)
    for _ in range(ITERS):
        indexes = residual_resample(weights)
        for i in indexes:
            counts[i] += 1

    means = counts / ITERS
    assert 2.55 < means[0] < 2.85   # expected 2.7
    assert 0.22 < means[1] < 0.32   # expected 0.27
    assert 0.02 < means[2] < 0.04   # expected 0.03
