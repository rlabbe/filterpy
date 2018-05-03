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

from filterpy.common import Saver
from filterpy.kalman import KalmanFilter, InformationFilter


DO_PLOT = False
def test_1d_0P():
    f = KalmanFilter (dim_x=2, dim_z=1)
    inf = InformationFilter (dim_x=2, dim_z=1)

    f.x = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    f.F = (np.array([[1.,1.],
                     [0.,1.]]))    # state transition matrix

    f.H = np.array([[1.,0.]])    # Measurement function
    f.R = np.array([[5.]])                 # state uncertainty
    f.Q = np.eye(2)*0.0001                 # process uncertainty
    f.P = np.diag([20., 20.])

    inf.x = f.x.copy()
    inf.F = f.F.copy()
    inf.H = np.array([[1.,0.]])    # Measurement function
    inf.R_inv *= 1./5                 # state uncertainty
    inf.Q *= 0.0001
    inf.P_inv = 0
    #inf.P_inv = inv(f.P)

    m = []
    r = []
    r2 = []


    zs = []
    for t in range (100):
        # create measurement = t plus white noise
        z = t + random.randn()*20
        zs.append(z)

        # perform kalman filtering
        f.predict()
        f.update(z)

        inf.predict()
        inf.update(z)

        # save data
        r.append (f.x[0,0])
        r2.append (inf.x[0,0])
        m.append(z)

        #assert abs(f.x[0,0] - inf.x[0,0]) < 1.e-12

    if DO_PLOT:
        plt.plot(m)
        plt.plot(r)
        plt.plot(r2)



def test_1d():
    f = KalmanFilter(dim_x=2, dim_z=1)
    inf = InformationFilter(dim_x=2, dim_z=1)

    # ensure __repr__ doesn't assert
    str(inf)


    f.x = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    inf.x = f.x.copy()
    f.F = (np.array([[1.,1.],
                     [0.,1.]]))    # state transition matrix

    inf.F = f.F.copy()
    f.H = np.array([[1.,0.]])      # Measurement function
    inf.H = np.array([[1.,0.]])    # Measurement function
    f.R *= 5                       # state uncertainty
    inf.R_inv *= 1./5               # state uncertainty
    f.Q *= 0.0001                  # process uncertainty
    inf.Q *= 0.0001


    m = []
    r = []
    r2 = []
    zs = []
    s = Saver(inf)
    for t in range (100):
        # create measurement = t plus white noise
        z = t + random.randn()*20
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        inf.update(z)
        inf.predict()

        # save data
        r.append (f.x[0,0])
        r2.append (inf.x[0,0])
        m.append(z)
        s.save()

        assert abs(f.x[0,0] - inf.x[0,0]) < 1.e-12


    if DO_PLOT:
        plt.plot(m)
        plt.plot(r)
        plt.plot(r2)


if __name__ == "__main__":
    DO_PLOT = True
    test_1d()
