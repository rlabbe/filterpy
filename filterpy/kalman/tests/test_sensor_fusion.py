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
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from numpy import array, asarray

DO_PLOT = False
SEED = 1124


def single_measurement_test():
    dt = 0.1
    sigma = 2.

    kf2 = KalmanFilter(dim_x=2, dim_z=1)

    kf2.F = array ([[1., dt], [0., 1.]])
    kf2.H = array ([[1., 0.]])
    kf2.x = array ([[0.], [1.]])
    kf2.Q = array ([[dt**3/3, dt**2/2],
                    [dt**2/2, dt]]) * 0.02
    kf2.P *= 100
    kf2.R[0,0] = sigma**2

    random.seed(SEED)
    xs = []
    zs = []
    nom = []
    for i in range(1, 100):
        m0 = i + randn()*sigma
        z = array([[m0]])

        kf2.predict()
        kf2.update(z)

        xs.append(kf2.x.T[0])
        zs.append(z.T[0])
        nom.append(i)

    xs = asarray(xs)
    zs = asarray(zs)
    nom = asarray(nom)


    res = nom-xs[:,0]
    std_dev = np.std(res)
    print('std: {:.3f}'.format (std_dev))

    global DO_PLOT
    if DO_PLOT:

        plt.subplot(211)
        plt.plot(xs[:,0])
        #plt.plot(zs[:,0])


        plt.subplot(212)
        plt.plot(res)
        plt.show()

    return std_dev


def sensor_fusion_test(wheel_sigma=2., gps_sigma=4.):
    dt = 0.1

    kf2 = KalmanFilter(dim_x=2, dim_z=2)

    kf2.F = array ([[1., dt], [0., 1.]])
    kf2.H = array ([[1., 0.], [1., 0.]])
    kf2.x = array ([[0.], [0.]])
    kf2.Q = array ([[dt**3/3, dt**2/2],
                    [dt**2/2, dt]]) * 0.02
    kf2.P *= 100
    kf2.R[0,0] = wheel_sigma**2
    kf2.R[1,1] = gps_sigma**2


    random.seed(SEED)
    xs = []
    zs = []
    nom = []
    for i in range(1, 100):
        m0 = i + randn()*wheel_sigma
        m1 = i + randn()*gps_sigma
        if gps_sigma >1e40:
            m1 = -1e40

        z = array([[m0], [m1]])

        kf2.predict()
        kf2.update(z)

        xs.append(kf2.x.T[0])
        zs.append(z.T[0])
        nom.append(i)

    xs = asarray(xs)
    zs = asarray(zs)
    nom = asarray(nom)


    res = nom-xs[:,0]
    std_dev = np.std(res)
    print('fusion std: {:.3f}'.format (np.std(res)))

    if DO_PLOT:

        plt.subplot(211)
        plt.plot(xs[:,0])
        #plt.plot(zs[:,0])
        #plt.plot(zs[:,1])

        plt.subplot(212)
        plt.axhline(0)
        plt.plot(res)
        plt.show()

    print(kf2.Q)
    print(kf2.K)
    return std_dev


def test_fusion():
    std1 = sensor_fusion_test()
    std2 = single_measurement_test()
    assert (std1 < std2)

if __name__ == "__main__":
    DO_PLOT=True
    sensor_fusion_test(2,4e100)
    single_measurement_test()

    test_fusion()
