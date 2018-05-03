# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


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

#pylint: skip-file

""" This is very old code, and no longer runs due to reorgization of the
UKF code"""


import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from GetRadar import get_radar
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise



def fx(x, dt):
    """ state transition function for sstate [downrange, vel, altitude]"""
    F = np.array([[1., dt, 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])

    return np.dot(F, x)


def hx(x):
    """ returns slant range based on downrange distance and altitude"""

    return (x[0]**2 + x[2]**2)**.5


if __name__ == "__main__":

    dt = 0.05

    radarUKF = UKF(dim_x=3, dim_z=1, dt=dt, kappa=0.)
    radarUKF.Q *= Q_discrete_white_noise(3, 1, .01)
    radarUKF.R *= 10
    radarUKF.x = np.array([0., 90., 1100.])
    radarUKF.P *= 100.

    t = np.arange(0, 20+dt, dt)
    n = len(t)
    xs = []
    rs = []
    for i in range(n):
        r = GetRadar(dt)
        rs.append(r)

        radarUKF.update(r, hx, fx)

        xs.append(radarUKF.x)

    xs = np.asarray(xs)

    plt.subplot(311)
    plt.plot(t, xs[:, 0])
    plt.title('distance')

    plt.subplot(312)
    plt.plot(t, xs[:, 1])
    plt.title('velocity')

    plt.subplot(313)
    plt.plot(t, xs[:, 2])
    plt.title('altitude')
