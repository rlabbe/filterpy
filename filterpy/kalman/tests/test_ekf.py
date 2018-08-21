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


import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, eye, asarray

from filterpy.common import Saver
from filterpy.examples import RadarSim
from pytest import approx
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

DO_PLOT = False


def test_ekf():
    def H_of(x):
        """ compute Jacobian of H matrix for state x """

        horiz_dist = x[0]
        altitude = x[2]

        denom = sqrt(horiz_dist**2 + altitude**2)
        return array([[horiz_dist/denom, 0., altitude/denom]])

    def hx(x):
        """ takes a state variable and returns the measurement that would
        correspond to that state.
        """

        return sqrt(x[0]**2 + x[2]**2)

    dt = 0.05
    proccess_error = 0.05

    rk = ExtendedKalmanFilter(dim_x=3, dim_z=1)

    rk.F = eye(3) + array ([[0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 0]])*dt

    def fx(x, dt):
        return np.dot(rk.F, x)

    rk.x = array([-10., 90., 1100.])
    rk.R *= 10
    rk.Q = array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]) * 0.001

    rk.P *= 50

    rs = []
    xs = []
    radar = RadarSim(dt)
    ps = []
    pos = []

    s = Saver(rk)
    for i in range(int(20/dt)):
        z = radar.get_range(proccess_error)
        pos.append(radar.pos)

        rk.update(asarray([z]), H_of, hx, R=hx(rk.x)*proccess_error)
        ps.append(rk.P)
        rk.predict()

        xs.append(rk.x)
        rs.append(z)
        s.save()

        # test mahalanobis
        a = np.zeros(rk.y.shape)
        maha = scipy_mahalanobis(a, rk.y, rk.SI)
        assert rk.mahalanobis == approx(maha)

    s.to_array()

    xs = asarray(xs)
    ps = asarray(ps)
    rs = asarray(rs)

    p_pos = ps[:, 0, 0]
    p_vel = ps[:, 1, 1]
    p_alt = ps[:, 2, 2]
    pos = asarray(pos)

    if DO_PLOT:
        plt.subplot(311)
        plt.plot(xs[:, 0])
        plt.ylabel('position')

        plt.subplot(312)
        plt.plot(xs[:, 1])
        plt.ylabel('velocity')

        plt.subplot(313)
        #plt.plot(xs[:,2])
        #plt.ylabel('altitude')

        plt.plot(p_pos)
        plt.plot(-p_pos)
        plt.plot(xs[:, 0] - pos)


if __name__ == '__main__':
    test_ekf()