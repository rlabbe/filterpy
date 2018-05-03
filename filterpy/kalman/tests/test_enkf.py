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

from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.common import Q_discrete_white_noise, Saver
from math import cos, sin

DO_PLOT = False

def test_1d_const_vel():

    def hx(x):
        return np.array([x[0]])

    F = np.array([[1., 1.],[0., 1.]])
    def fx(x, dt):
        return np.dot(F, x)

    x = np.array([0., 1.])
    P = np.eye(2)* 100.
    f = EnKF(x=x, P=P, dim_z=1, dt=1., N=8, hx=hx, fx=fx)

    std_noise = 10.

    f.R *= std_noise**2
    f.Q = Q_discrete_white_noise(2, 1., .001)

    measurements = []
    results = []
    ps = []
    zs = []
    s = Saver(f)
    for t in range (0,100):
        # create measurement = t plus white noise
        z = t + randn()*std_noise
        zs.append(z)

        f.predict()
        f.update(np.asarray([z]))

        # save data
        results.append (f.x[0])
        measurements.append(z)
        ps.append(3*(f.P[0,0]**.5))
        s.save()
    s.to_array()

    results = np.asarray(results)
    ps = np.asarray(ps)

    if DO_PLOT:
        plt.plot(results, label='EnKF')
        plt.plot(measurements, c='r', label='z')
        plt.plot (results-ps, c='k',linestyle='--', label='3$\sigma$')
        plt.plot(results+ps, c='k', linestyle='--')
        plt.legend(loc='best')
        #print(ps)
    return f



def test_circle():
    def hx(x):
        return np.array([x[0], x[3]])

    F = np.array([[1., 1., .5, 0., 0., 0.],
                  [0., 1., 1., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 1., 1., .5],
                  [0., 0., 0., 0., 1., 1.],
                  [0., 0., 0., 0., 0., 1.]])

    def fx(x, dt):
        return np.dot(F, x)

    x = np.array([50., 0., 0, 0, .0, 0.])
    P = np.eye(6)* 100.
    f = EnKF(x=x, P=P, dim_z=2, dt=1., N=30, hx=hx, fx=fx)

    std_noise = .1

    f.R *= std_noise**2
    f.Q[0:3, 0:3] = Q_discrete_white_noise(3, 1., .001)
    f.Q[3:6, 3:6] = Q_discrete_white_noise(3, 1., .001)

    measurements = []
    results = []

    zs = []
    for t in range (0,300):
        a = t / 300000
        x = cos(a) * 50.
        y = sin(a) * 50.
        # create measurement = t plus white noise
        z = np.array([x,y])
        zs.append(z)

        f.predict()
        f.update(z)

        # save data
        results.append (f.x)
        measurements.append(z)

    #test that __repr__ doesn't assert
    str(f)

    results = np.asarray(results)
    measurements = np.asarray(measurements)

    if DO_PLOT:
        plt.plot(results[:,0], results[:,2], label='EnKF')
        plt.plot(measurements[:,0], measurements[:,1], c='r', label='z')
        #plt.plot (results-ps, c='k',linestyle='--', label='3$\sigma$')
        #plt.plot(results+ps, c='k', linestyle='--')
        plt.legend(loc='best')
        plt.axis('equal')
        #print(ps)






if __name__ == '__main__':
    DO_PLOT = True
    test_circle ()
    test_1d_const_vel()


#test_noisy_1d()