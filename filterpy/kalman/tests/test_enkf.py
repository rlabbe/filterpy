# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 09:10:29 2014

@author: rlabbe
"""

from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.common import Q_discrete_white_noise


DO_PLOT = False

def test_1d_const_vel():

    def hx(x):
        return np.array([x[0]])

    F = np.array([[1., 1.],[0., 1.]])
    def fx(x, dt):
        return np.array([x[0]+x[1], x[1]])

    x = np.array([0., 1.])
    P = np.eye(2)* 100.
    f = EnKF(x=x, P=P, dim_z=1, dt=1., N=8, hx=hx, fx=fx)


    std_noise = 3.

    f.R *= std_noise**2
    f.Q = Q_discrete_white_noise(2, 1., .001)

    f.initialize (f.x, f.P)

    measurements = []
    results = []
    ps = []

    zs = []
    for t in range (0,100):
        # create measurement = t plus white noise
        z = t + randn()*std_noise
        zs.append(z)

        f.predict()
        f.update(np.asarray([z]))

        # save data
        results.append (f.x[0])
        measurements.append(z)
        ps.append(f.P[0,0]**.5)
        print()

    results = np.asarray(results)
    ps = np.asarray(ps)

    if DO_PLOT:
        plt.plot(results, label='EnKF')
        plt.plot(measurements, c='r', label='z')
        plt.plot (results-ps, c='k',linestyle='--')
        plt.plot(results+ps, c='k', linestyle='--')
        plt.legend(loc='best')
        #print(ps)



if __name__ == '__main__':
    DO_PLOT = True

    test_1d_const_vel()


#test_noisy_1d()