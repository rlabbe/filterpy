# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 08:58:05 2014

@author: rlabbe
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy.random as random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from filterpy.kalman import KalmanFilter, InformationFilter


DO_PLOT = False
def test_1d_0P():
    f = KalmanFilter (dim_x=2, dim_z=1)
    inf = InformationFilter (dim_x=2, dim_z=1)

    f.X = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    inf.X = f.X.copy()
    f.F = (np.array([[1.,1.],
                     [0.,1.]]))    # state transition matrix

    inf.F = f.F.copy()
    f.H = np.array([[1.,0.]])    # Measurement function
    inf.H = np.array([[1.,0.]])    # Measurement function
    f.R = 5.                 # state uncertainty
    inf.R_inv = 1./5                 # state uncertainty
    f.Q = 0.0001                 # process uncertainty
    inf.Q = 0.0001
    f.P *= 20
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
        r.append (f.X[0,0])
        r2.append (inf.X[0,0])
        m.append(z)

        #assert abs(f.X[0,0] - inf.X[0,0]) < 1.e-12

    if DO_PLOT:
        plt.plot(m)
        plt.plot(r)
        plt.plot(r2)



def test_1d():
    f = KalmanFilter (dim_x=2, dim_z=1)
    inf = InformationFilter (dim_x=2, dim_z=1)

    f.X = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    inf.X = f.X.copy()
    f.F = (np.array([[1.,1.],
                     [0.,1.]]))    # state transition matrix

    inf.F = f.F.copy()
    f.H = np.array([[1.,0.]])      # Measurement function
    inf.H = np.array([[1.,0.]])    # Measurement function
    f.R = 5.                       # state uncertainty
    inf.R_inv = 1./5               # state uncertainty
    f.Q = 0.0001                   # process uncertainty
    inf.Q = 0.0001

    m = []
    r = []
    r2 = []


    zs = []
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
        r.append (f.X[0,0])
        r2.append (inf.X[0,0])
        m.append(z)

        assert abs(f.X[0,0] - inf.X[0,0]) < 1.e-12

    if DO_PLOT:
        plt.plot(m)
        plt.plot(r)
        plt.plot(r2)


if __name__ == "__main__":
    DO_PLOT = True
    test_1d()