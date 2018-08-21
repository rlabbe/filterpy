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

import numpy.random as random
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import FadingKalmanFilter
from pytest import approx
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

DO_PLOT = False
def test_noisy_1d():
    f = FadingKalmanFilter(3., dim_x=2, dim_z=1)

    f.x = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    f.F = np.array([[1.,1.],
                    [0.,1.]])    # state transition matrix

    f.H = np.array([[1.,0.]])     # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R = 5.**2                    # state uncertainty
    f.Q = np.array([[0, 0],
                    [0, 0.0001]]) # process uncertainty

    measurements = []
    results = []

    zs = []
    for t in range (100):
        # create measurement = t plus white noise
        z = t + random.randn() * np.sqrt(f.R)
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        # save data
        results.append(f.x[0, 0])
        measurements.append(z)

        # test mahalanobis
        a = np.zeros(f.y.shape)
        maha = scipy_mahalanobis(a, f.y, f.SI)
        assert f.mahalanobis == approx(maha)
        print(z, maha, f.y, f.S)
        assert maha < 4


    # now do a batch run with the stored z values so we can test that
    # it is working the same as the recursive implementation.
    # give slightly different P so result is slightly different
    f.X = np.array([[2.,0]]).T
    f.P = np.eye(2)*100.
    m, c, _, _ = f.batch_filter(zs,update_first=False)

    # plot data
    if DO_PLOT:
        p1, = plt.plot(measurements,'r', alpha=0.5)
        p2, = plt.plot (results,'b')
        p4, = plt.plot(m[:,0], 'm')
        p3, = plt.plot ([0, 100],[0, 100], 'g') # perfect result
        plt.legend([p1,p2, p3, p4],
                   ["noisy measurement", "KF output", "ideal", "batch"], loc=4)


        plt.show()


if __name__ == "__main__":
    DO_PLOT = True
    test_noisy_1d()