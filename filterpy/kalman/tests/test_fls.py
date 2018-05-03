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
from filterpy.kalman import KalmanFilter, FixedLagSmoother




DO_PLOT = False


def test_fls():

    # it is possible for the fixed lag to rarely perform worse than the
    # kalman filter. Let it happen once in 50 times before we become
    # alarmed.

    fail_count = 0
    for i in range(50):
        fail_count = one_run_test_fls()

    assert fail_count < 2



def test_batch_equals_recursive():
    """ ensures that the batch filter and the recursive version both
    produce the same results.
    """

    N = 4 # size of lag

    fls = FixedLagSmoother(dim_x=2, dim_z=1, N=N)

    fls.x = np.array([0., .5])

    fls.F = np.array([[1.,1.],
                      [0.,1.]])

    fls.H = np.array([[1.,0.]])

    fls.P *= 200
    fls.R *= 5.
    fls.Q *= 0.001


    nom =  np.array([t/2. for t in range (0,40)])
    zs = np.array([t + random.randn()*1.1 for t in nom])

    xs, x = fls.smooth_batch(zs, N)


    for k,z in enumerate(zs):
        fls.smooth(z)

    xSmooth = np.asarray(fls.xSmooth)
    xfl = xs[:,0].T[0]

    res = xSmooth.T[0,0] - xfl

    assert np.sum(res) < 1.e-12





def one_run_test_fls():
    fls = FixedLagSmoother(dim_x=2, dim_z=1)

    fls.x = np.array([0., .5])
    fls.F = np.array([[1.,1.],
                      [0.,1.]])

    fls.H = np.array([[1.,0.]])
    fls.P *= 200
    fls.R *= 5.
    fls.Q *= 0.001

    kf = KalmanFilter(dim_x=2, dim_z=1)

    kf.x = np.array([0., .5])
    kf.F = np.array([[1.,1.],
                     [0.,1.]])
    kf.H = np.array([[1.,0.]])
    kf.P *= 2000
    kf.R *= 1.
    kf.Q *= 0.001

    N = 4 # size of lag

    nom =  np.array([t/2. for t in range (0,40)])
    zs = np.array([t + random.randn()*1.1 for t in nom])

    xs, x = fls.smooth_batch(zs, N)

    M, P, _, _ = kf.batch_filter(zs)
    rts_x, _, _, _ = kf.rts_smoother(M, P)

    xfl = xs[:,0].T[0]
    xkf = M[:,0].T[0]

    fl_res = abs(xfl-nom)
    kf_res = abs(xkf-nom)

    if DO_PLOT:
        plt.cla()
        plt.plot(zs,'o', alpha=0.5, marker='o', label='zs')
        plt.plot(x[:,0], label='FLS')
        plt.plot(xfl, label='FLS S')
        plt.plot(xkf, label='KF')
        plt.plot(rts_x[:,0], label='RTS')
        plt.legend(loc=4)
        plt.show()


        print(fl_res)
        print(kf_res)

        print('std fixed lag:', np.mean(fl_res[N:]))
        print('std kalman:', np.mean(kf_res[N:]))

    return np.mean(fl_res) <= np.mean(kf_res)


if __name__ == '__main__':
    DO_PLOT = True


    one_run_test_fls()

    DO_PLOT = False
    test_fls()

    test_batch_equals_recursive()
