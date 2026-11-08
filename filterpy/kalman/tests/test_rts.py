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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


DO_PLOT = False

def test_rts():
    fk = KalmanFilter(dim_x=2, dim_z=1)

    fk.x = np.array([-1., 1.])    # initial state (location and velocity)

    fk.F = np.array([[1.,1.],
                     [0.,1.]])      # state transition matrix

    fk.H = np.array([[1.,0.]])      # Measurement function
    fk.P = .01                     # covariance matrix
    fk.R = 5                       # state uncertainty
    fk.Q = 0.001                   # process uncertainty


    zs = [t + random.randn()*4 for t in range(40)]

    mu, cov, _, _ = fk.batch_filter (zs)
    mus = [x[0] for x in mu]

    M, P, _, _ = fk.rts_smoother(mu, cov)

    if DO_PLOT:
        p1, = plt.plot(zs,'cyan', alpha=0.5)
        p2, = plt.plot(M[:,0],c='b')
        p3, = plt.plot(mus,c='r')
        p4, = plt.plot([0, len(zs)], [0, len(zs)], 'g') # perfect result
        plt.legend([p1, p2, p3, p4],
                   ["measurement", "RKS", "KF output", "ideal"], loc=4)

        plt.show()



def test_rts_control_input():
    """rts_smoother() must undo the same B@u term predict() applied, or a
    constant control input biases every smoothed estimate by that term.
    See issue #164: there was no Bs/us parameter at all, so the smoother
    always used the (buggy) uncontrolled dynamics for its correction step.
    """
    fk = KalmanFilter(dim_x=1, dim_z=1)
    fk.F = np.array([[1.]])
    fk.Q = np.array([[0.1]])

    u = 5.
    n = 5
    # Xs is exactly the trajectory x[k+1] = F@x[k] + B@u would produce, so a
    # smoother that correctly accounts for the control input has nothing
    # left to correct: the innovation at every step is exactly zero.
    Xs = np.array([[u * k] for k in range(n)])
    Ps = np.array([np.eye(1) for _ in range(n)])
    Bs = [np.array([[1.]])] * n
    us = [np.array([u])] * n

    x_with_control, _, _, _ = fk.rts_smoother(Xs, Ps, Bs=Bs, us=us)
    assert np.allclose(x_with_control, Xs), \
        "a trajectory that exactly matches F@x + B@u should need no smoothing correction"

    # Bs/us are optional: omitting them keeps the pre-fix, uncontrolled
    # formula, which does not recognise this trajectory as consistent and
    # so still perturbs it -- confirming the old call signature still works
    # and that the two code paths genuinely differ.
    x_without_control, _, _, _ = fk.rts_smoother(Xs, Ps)
    assert not np.allclose(x_without_control, Xs)


if __name__ == '__main__':
    DO_PLOT = True
    test_rts()
    test_rts_control_input()

