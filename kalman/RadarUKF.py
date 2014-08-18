# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import SigmaPoints as ukf
from GetRadar import *
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def fx(x, dt):
    A = np.eye(3) + dt * np.array ([[0, 1, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
    return A * x


def hx(x):
    return np.sqrt (x[0]**2 + x[2]**2)


def RadarUKF (z, dt):
    if not hasattr (RadarUKF, "Q"):
        RadarUKF.Q = 0.01 * np.eye(3)
        RadarUKF.R = 100
        RadarUKF.x = np.array ([[0.,90., 1100.]]).T
        RadarUKF.P = np.eye(3) * 100.
        RadarUKF.n = 3
        RadarUKF.m = 1

    Xi, W = ukf.sigma_points (RadarUKF.x, RadarUKF.P, 0)
    print(Xi, type(Xi))

    fXi = np.zeros ((RadarUKF.n, 2*RadarUKF.n+1))
    print(fXi)

    a = fx(Xi[:,0], dt)
    print('a=',a)
    print ('fxi=',fXi[:,0])

    for i in range (2*RadarUKF.n+1):
        print(i)
        fXi[:,i] = fx(Xi[:,i], dt)


    xp, Pp = ukf.unscented_transform (fXi, W, RadarUKF.Q)

    hXi = np.zeros((RadarUKF.m, 2*RadarUKF.n+1))
    for i in range (2*RadarUKF.n+1):
        hXi[:, i] = hx(fXi[:,i])

    zp, Pz = ukf.unscented_transform(hXi, W, RadarUKF.R)
    Pxz = np.zeros((RadarUKF.n,RadarUKF.m))

    for i in range (2*RadarUKF.n+1):
        Pxz = Pxz + W[i] * (fXi[:,i] - xp) * (hXi[:,i] - zp).T

    K = Pxz * linalg.inv(Pz)

    RadarUKF.x = xp + K * (z - zp)
    RadarUKF.P = Pp - K * Pz * K.T

    return RadarUKF.x

if __name__ == "__main__":
    dt = 0.05
    t = np.arange (0,20+dt, dt)
    n = len(t)
    print('n=', n)


    x = np.zeros((3, n))
    rs = []


    for i in range(n):
        r = GetRadar(dt)
        #r = 991.95
        rs.append(r)
        q = RadarUKF(r, dt)
        x[:,i] = q
        #print 'q=', q

    plt.figure(1)
    plt.plot(t, x.A[0,:])

    plt.figure(2)
    plt.plot(t, x.A[1,:])

    plt.figure (3)
    plt.plot(t, x.A[2,:])



