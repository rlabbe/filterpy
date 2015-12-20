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
from filterpy.kalman import KalmanFilter, FixedPointSmoother, GrewalFixedPointSmoother
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

DO_PLOT = False


class PosSensor1(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]

    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        return [self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std]


def test_fixed_point():
    j = 8
    kf = FixedPointSmoother(4, 2, j)

    dt = 1.
    z_std = .5
    kf.x = np.array([[20., 1, 20, 1]]).T
    kf.P *= 500.
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])

    kf.H = np.array([[1., 0, 0,  0],
                     [0., 0, 1, 0]])

    kf.R *= z_std**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
    kf.Q = block_diag(q, q)


    xs = np.linspace(0, 39, 40)
    ys = np.linspace(0, 39, 40)
    zs = np.array([np.array([[x+randn()*z_std, y+randn()*z_std]]).T for x, y in zip(xs, ys)])
    est, estj = [], []
    Ps = []
    for z in zs:

        kf.smooth(z)
        est.append(kf.x.copy())
        try:
            estj.append(kf.xj.copy())
            #print(kf.x.T, kf.xj.T)
        except:
            pass

    if DO_PLOT:
        est = np.array(est)
        estj = np.array(estj)
        plt.subplot(121)
        plt.plot(est[:, 0], est[:, 2])
        plt.subplot(122)
        plt.plot(estj[:, 0])
        plt.gca().axhline(est[j][0])
        plt.gca().axhline(j, color='k')
        #print(estj)
        #print(est[j])
        plt.show()


def test_fixed_point1d():
    j = 8
    kf = GrewalFixedPointSmoother(2, 1, j)

    dt = 1.
    z_std = .5
    kf.x = np.array([[20., 1]]).T
    kf.P *= 500.
    kf.F = np.array([[1, dt],
                     [0, 1]])

    kf.H = np.array([[1., 0]])

    kf.R *= z_std**2
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)


    xs = np.linspace(0, 39, 40)
    zs = [x+randn()*z_std for x in xs]
    est, estj = [], []
    for z in zs:
        kf.smooth(z)
        est.append(kf.x.copy())
        try:
            estj.append(kf.xj.copy())
            #print(kf.x.T, kf.xj.T)
        except:
            pass

    if DO_PLOT:
        est = np.array(est)
        estj = np.array(estj)
        plt.subplot(121)
        #plt.plot(est[:, 0], est[:, 2])
        plt.subplot(122)
        plt.plot(estj[:, 0])
        plt.gca().axhline(est[j][0])
        plt.gca().axhline(j, color='k')
        #print(estj)
        #print(est[j])
        plt.show()



def test_noisy_1d():
    f = KalmanFilter (dim_x=2, dim_z=1)

    f.x = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    f.F = np.array([[1.,1.],
                    [0.,1.]])    # state transition matrix

    f.H = np.array([[1.,0.]])    # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R = 5                       # state uncertainty
    f.Q = 0.0001                 # process uncertainty

    measurements = []
    results = []

    zs = []
    for t in range (100):
        # create measurement = t plus white noise
        z = t + random.randn()*20
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        # save data
        results.append (f.x[0,0])
        measurements.append(z)


    # now do a batch run with the stored z values so we can test that
    # it is working the same as the recursive implementation.
    # give slightly different P so result is slightly different
    f.x = np.array([[2.,0]]).T
    f.P = np.eye(2)*100.
    m,c,_,_ = f.batch_filter(zs,update_first=False)

    # plot data
    if DO_PLOT:
        p1, = plt.plot(measurements,'r', alpha=0.5)
        p2, = plt.plot (results,'b')
        p4, = plt.plot(m[:,0], 'm')
        p3, = plt.plot ([0,100],[0,100], 'g') # perfect result
        plt.legend([p1,p2, p3, p4],
                   ["noisy measurement", "KF output", "ideal", "batch"], loc=4)


        plt.show()


def test_1d_vel():
    from scipy.linalg import inv
    global ks
    dt = 1.
    std_z = 0.0001

    x = np.array([[0.], [0.]])

    F = np.array([[1., dt],
                    [0., 1.]])

    H = np.array([[1.,0.]])
    P = np.eye(2)
    R = np.eye(1)*std_z**2
    Q = np.eye(2)*0.001

    measurements = []
    results = []

    xest = []
    ks = []
    pos = 0.
    for t in range (20):
        z = pos + random.randn() * std_z
        pos += 100

        # perform kalman filtering
        x = F @ x
        P = F @ P @ F.T + Q

        P2 = P.copy()
        P2[0,1] = 0 # force there to be no correlation
        P2[1,0] = 0
        S = H @ P2 @ H.T + R
        K = P2 @ H.T @inv(S)
        y = z - H@x
        x = x + K@y

        # save data
        xest.append (x.copy())
        measurements.append(z)
        ks.append(K.copy())

    xest = np.array(xest)
    ks = np.array(ks)
    # plot data
    if DO_PLOT:
        plt.subplot(121)
        plt.plot(xest[:, 1])
        plt.subplot(122)
        plt.plot(ks[:, 1])
        plt.show()



def test_noisy_11d():
    f = KalmanFilter (dim_x=2, dim_z=1)

    f.x = np.array([2., 0])      # initial state (location and velocity)

    f.F = np.array([[1.,1.],
                    [0.,1.]])    # state transition matrix

    f.H = np.array([[1.,0.]])    # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R = 5                       # state uncertainty
    f.Q = 0.0001                 # process uncertainty

    measurements = []
    results = []

    zs = []
    for t in range (100):
        # create measurement = t plus white noise
        z = t + random.randn()*20
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        # save data
        results.append (f.x[0])
        measurements.append(z)


    # now do a batch run with the stored z values so we can test that
    # it is working the same as the recursive implementation.
    # give slightly different P so result is slightly different
    f.x = np.array([[2.,0]]).T
    f.P = np.eye(2)*100.
    m,c,_,_ = f.batch_filter(zs,update_first=False)

    # plot data
    if DO_PLOT:
        p1, = plt.plot(measurements,'r', alpha=0.5)
        p2, = plt.plot (results,'b')
        p4, = plt.plot(m[:,0], 'm')
        p3, = plt.plot ([0,100],[0,100], 'g') # perfect result
        plt.legend([p1,p2, p3, p4],
                   ["noisy measurement", "KF output", "ideal", "batch"], loc=4)

        plt.show()


def test_batch_filter():
    f = KalmanFilter (dim_x=2, dim_z=1)

    f.x = np.array([2., 0])      # initial state (location and velocity)

    f.F = np.array([[1.,1.],
                    [0.,1.]])    # state transition matrix

    f.H = np.array([[1.,0.]])    # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R = 5                       # state uncertainty
    f.Q = 0.0001                 # process uncertainty

    zs = [None, 1., 2.]
    m,c,_,_ = f.batch_filter(zs,update_first=False)
    m,c,_,_ = f.batch_filter(zs,update_first=True)

def test_univariate():
    f = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
    f.x = np.array([[0]])
    f.P *= 50
    print(f.P)
    f.H = np.array([[1.]])
    f.F = np.array([[1.]])
    f.B = np.array([[1.]])
    f.Q = .02
    f.R *= .1

    for i in range(50):
        f.predict();
        f.update(i)


if __name__ == "__main__":
    DO_PLOT = True
    test_1d_vel()
    #test_batch_filter()

    #test_univariate()
    #test_noisy_11d()