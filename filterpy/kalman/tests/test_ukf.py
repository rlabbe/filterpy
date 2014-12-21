# -*- coding: utf-8 -*-

"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
6"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import matplotlib.pyplot as plt
import numpy.random as random
from numpy.random import randn
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import ScaledUnscentedKalmanFilter as SUKF
from filterpy.kalman import unscented_transform
from filterpy.common import stats, Q_discrete_white_noise
from math import cos, sin


DO_PLOT = False


def test_sigma_plot():
    """ Test to make sure sigma's correctly mirror the shape and orientation
    of the covariance array."""

    x = np.array([[1, 2]])
    P = np.array([[2, 1.2],
                  [1.2, 2]])
    kappa = .1

    # if kappa is larger, than points shoudld be closer together
    sp0 = UKF.weights(2, kappa)
    sp1 = UKF.weights(2, kappa*1000)

    Xi0 = UKF.sigma_points (x, P, kappa)
    Xi1 = UKF.sigma_points (x, P, kappa*1000)

    assert max(Xi1[:,0]) > max(Xi0[:,0])
    assert max(Xi1[:,1]) > max(Xi0[:,1])

    if DO_PLOT:
        plt.figure()
        for i in range(Xi0.shape[0]):
            plt.scatter((Xi0[i,0]-x[0, 0])*sp0[i] + x[0, 0],
                        (Xi0[i,1]-x[0, 1])*sp0[i] + x[0, 1],
                         color='blue')

        for i in range(Xi1.shape[0]):
            plt.scatter((Xi1[i, 0]-x[0, 0]) * sp1[i] + x[0,0],
                        (Xi1[i, 1]-x[0, 1]) * sp1[i] + x[0,1],
                         color='green')

        stats.plot_covariance_ellipse([1, 2], P)


def test_julier_weights():
    for n in range(1,15):
        for k in np.linspace(0,5,0.1):
            Wm = UKF.weights(n, k)

            assert abs(sum(Wm) - 1) < 1.e-12


def test_scaled_weights():
    for n in range(1,5):
        for alpha in np.linspace(0.99, 1.01, 100):
            for beta in range(0,2):
                for kappa in range(0,2):
                    Wm, Wc = SUKF.weights(n, alpha, 0, 3-n)
                    assert abs(sum(Wm) - 1) < 1.e-1
                    assert abs(sum(Wc) - 1) < 1.e-1


def test_sigma_points_1D():
    """ tests passing 1D data into sigma_points"""

    kappa = 0.
    ukf = UKF(dim_x=1, dim_z=1, dt=0.1, hx=None, fx=None, kappa=kappa)

    points = ukf.weights(1, 0.)
    assert len(points) == 3

    mean = 5
    cov = 9

    Xi = ukf.sigma_points (mean, cov, kappa)
    xm, ucov = unscented_transform(Xi, ukf.W, ukf.W, 0)

    # sum of weights*sigma points should be the original mean
    m = 0.0
    for x,w in zip(Xi, ukf.W):
        m += x*w

    assert abs(m-mean) < 1.e-12
    assert abs(xm[0] - mean) < 1.e-12
    assert abs(ucov[0,0]-cov) < 1.e-12

    assert Xi.shape == (3,1)
    assert len(ukf.W) == 3


class RadarSim(object):
    def __init__(self, dt):
        self.x = 0
        self.dt = dt

    def get_range(self):
        vel = 100  + 5*randn()
        alt = 1000 + 10*randn()
        self.x += vel*self.dt

        v = self.x * 0.05*randn()
        rng = (self.x**2 + alt**2)**.5 + v
        return rng


def test_radar():
    def fx(x, dt):
        A = np.eye(3) + dt * np.array ([[0, 1, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]])
        return A.dot(x)

    def hx(x):
        return np.sqrt (x[0]**2 + x[2]**2)

    dt = 0.05

    kf = UKF(3, 1, dt, fx=fx, hx=hx, kappa=0.)

    kf.Q *= 0.01
    kf.R = 10
    kf.x = np.array([0., 90., 1100.])
    kf.P *= 100.
    radar = RadarSim(dt)

    t = np.arange(0,20+dt, dt)

    n = len(t)

    xs = np.zeros((n,3))

    random.seed(200)
    rs = []
    #xs = []
    for i in range(len(t)):
        r = radar.get_range()
        #r = GetRadar(dt)
        kf.predict()
        kf.update(z=[r])

        xs[i,:] = kf.x
        rs.append(r)

    if DO_PLOT:
        print(xs[:,0].shape)

        plt.figure()
        plt.subplot(311)
        plt.plot(t, xs[:,0])
        plt.subplot(312)
        plt.plot(t, xs[:,1])
        plt.subplot(313)

        plt.plot(t, xs[:,2])



def test_rts():
    def fx(x, dt):
        A = np.eye(3) + dt * np.array ([[0, 1, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]])
        f = np.dot(A, x)
        return f

    def hx(x):
        return np.sqrt (x[0]**2 + x[2]**2)

    dt = 0.05

    kf = UKF(3, 1, dt, fx=fx, hx=hx, kappa=1.)

    kf.Q *= 0.01
    kf.R = 10
    kf.x = np.array([0., 90., 1100.])
    kf.P *= 100.
    radar = RadarSim(dt)

    t = np.arange(0,20+dt, dt)

    n = len(t)

    xs = np.zeros((n,3))

    random.seed(200)
    rs = []
    #xs = []
    for i in range(len(t)):
        r = radar.get_range()
        #r = GetRadar(dt)
        kf.predict()
        kf.update(z=[r])

        xs[i,:] = kf.x
        rs.append(r)


    kf.x = np.array([0., 90., 1100.])
    kf.P = np.eye(3)*100
    M, P = kf.batch_filter(rs)
    assert np.array_equal(M, xs), "Batch filter generated different output"

    Qs = [kf.Q]*len(t)
    M2, P2, K = kf.rts_smoother(Xs=M, Ps=P, Qs=Qs)


    if DO_PLOT:
        print(xs[:,0].shape)

        plt.figure()
        plt.subplot(311)
        plt.plot(t, xs[:,0])
        plt.plot(t, M2[:,0], c='g')
        plt.subplot(312)
        plt.plot(t, xs[:,1])
        plt.plot(t, M2[:,1], c='g')
        plt.subplot(313)

        plt.plot(t, xs[:,2])
        plt.plot(t, M2[:,2], c='g')


def test_fixed_lag():
    def fx(x, dt):
        A = np.eye(3) + dt * np.array ([[0, 1, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]])
        f = np.dot(A, x)
        return f

    def hx(x):
        return np.sqrt (x[0]**2 + x[2]**2)

    dt = 0.05

    kf = UKF(3, 1, dt, fx=fx, hx=hx, kappa=0.)

    kf.Q *= 0.01
    kf.R = 10
    kf.x = np.array([0., 90., 1100.])
    kf.P *= 1.
    radar = RadarSim(dt)

    t = np.arange(0,20+dt, dt)

    n = len(t)

    xs = np.zeros((n,3))

    random.seed(200)
    rs = []
    #xs = []

    M = []
    P = []
    N =10
    flxs = []
    for i in range(len(t)):
        r = radar.get_range()
        #r = GetRadar(dt)
        kf.predict()
        kf.update(z=[r])

        xs[i,:] = kf.x
        flxs.append(kf.x)
        rs.append(r)
        M.append(kf.x)
        P.append(kf.P)
        print(i)
        #print(i, np.asarray(flxs)[:,0])
        if i == 20 and len(M) >= N:
            try:
                M2, P2, K = kf.rts_smoother(Xs=np.asarray(M)[-N:], Ps=np.asarray(P)[-N:])
                flxs[-N:] = M2
                #flxs[-N:] = [20]*N
            except:
                print('except', i)
            #P[-N:] = P2


    kf.x = np.array([0., 90., 1100.])
    kf.P = np.eye(3)*100
    M, P = kf.batch_filter(rs)

    Qs = [kf.Q]*len(t)
    M2, P2, K = kf.rts_smoother(Xs=M, Ps=P, Qs=Qs)


    flxs = np.asarray(flxs)
    print(xs[:,0].shape)

    plt.figure()
    plt.subplot(311)
    plt.plot(t, xs[:,0])
    plt.plot(t, flxs[:,0], c='r')
    plt.plot(t, M2[:,0], c='g')
    plt.subplot(312)
    plt.plot(t, xs[:,1])
    plt.plot(t, flxs[:,1], c='r')
    plt.plot(t, M2[:,1], c='g')

    plt.subplot(313)
    plt.plot(t, xs[:,2])
    plt.plot(t, flxs[:,2], c='r')
    plt.plot(t, M2[:,2], c='g')



def test_circle():
    from filterpy.kalman import KalmanFilter
    from math import radians
    def hx(x):
        radius = x[0]
        angle = x[1]
        x = cos(radians(angle)) * radius
        y = sin(radians(angle)) * radius
        return np.array([x, y])

    def fx(x, dt):
        return np.array([x[0], x[1]+x[2], x[2]])

    std_noise = .1


    f = UKF(dim_x=3, dim_z=2, dt=.01, hx=hx, fx=fx, kappa=0)
    f.x = np.array([50., 90., 0])
    f.P *= 100
    f.R = np.eye(2)*(std_noise**2)
    f.Q = np.eye(3)*.001
    f.Q[0,0]=0
    f.Q[2,2]=0

    kf = KalmanFilter(dim_x=6, dim_z=2)
    kf.x = np.array([50., 0., 0, 0, .0, 0.])

    F = np.array([[1., 1., .5, 0., 0., 0.],
                  [0., 1., 1., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 1., 1., .5],
                  [0., 0., 0., 0., 1., 1.],
                  [0., 0., 0., 0., 0., 1.]])

    kf.F = F
    kf.P*= 100
    kf.H = np.array([[1,0,0,0,0,0],
                     [0,0,0,1,0,0]])


    kf.R = f.R
    kf.Q[0:3, 0:3] = Q_discrete_white_noise(3, 1., .00001)
    kf.Q[3:6, 3:6] = Q_discrete_white_noise(3, 1., .00001)

    measurements = []
    results = []

    zs = []
    kfxs = []
    for t in range (0,12000):
        a = t / 30 + 90
        x = cos(radians(a)) * 50.+ randn() * std_noise
        y = sin(radians(a)) * 50. + randn() * std_noise
        # create measurement = t plus white noise
        z = np.array([x,y])
        zs.append(z)

        f.predict()
        f.update(z)

        kf.predict()
        kf.update(z)

        # save data
        results.append (hx(f.x))
        kfxs.append(kf.x)
        print(f.x)

    results = np.asarray(results)
    zs = np.asarray(zs)
    kfxs = np.asarray(kfxs)

    print(results)
    if DO_PLOT:
        plt.plot(zs[:,0], zs[:,1], c='r', label='z')
        plt.plot(results[:,0], results[:,1], c='k', label='UKF')
        plt.plot(kfxs[:,0], kfxs[:,3], c='g', label='KF')
        plt.legend(loc='best')
        plt.axis('equal')



def kf_circle():
    from filterpy.kalman import KalmanFilter
    from math import radians
    import math
    def hx(x):
        radius = x[0]
        angle = x[1]
        x = cos(radians(angle)) * radius
        y = sin(radians(angle)) * radius
        return np.array([x, y])

    def fx(x, dt):
        return np.array([x[0], x[1]+x[2], x[2]])


    def hx_inv(x, y):
        angle = math.atan2(y,x)
        radius = math.sqrt(x*x + y*y)
        return np.array([radius, angle])


    std_noise = .1


    kf = KalmanFilter(dim_x=3, dim_z=2)
    kf.x = np.array([50., 0., 0, 0, .0, 0.])

    F = np.array([[1., 0, 0.],
                  [0., 1., 1.,],
                  [0., 0., 1.,]])

    kf.F = F
    kf.P*= 100
    kf.H = np.array([[1,0,0],
                     [0,1,0]])

    kf.R = np.eye(2)*(std_noise**2)
    #kf.Q[0:3, 0:3] = Q_discrete_white_noise(3, 1., .00001)



    zs = []
    kfxs = []
    for t in range (0,2000):
        a = t / 30 + 90
        x = cos(radians(a)) * 50.+ randn() * std_noise
        y = sin(radians(a)) * 50. + randn() * std_noise

        z = hx_inv(x,y)
        zs.append(z)

        kf.predict()
        kf.update(z)

        # save data
        kfxs.append(kf.x)

    zs = np.asarray(zs)
    kfxs = np.asarray(kfxs)


    if DO_PLOT:
        plt.plot(zs[:,0], zs[:,1], c='r', label='z')
        plt.plot(kfxs[:,0], kfxs[:,3], c='g', label='KF')
        plt.legend(loc='best')
        plt.axis('equal')



if __name__ == "__main__":

    #test_sigma_points_1D()
    #test_fixed_lag()
    #DO_PLOT = True
    test_rts()
    #kf_circle()
    #test_circle()


    '''test_1D_sigma_points()
    #plot_sigma_test ()

    x = np.array([[1,2]])
    P = np.array([[2, 1.2],
                  [1.2, 2]])
    kappa = .1

    xi,w = sigma_points (x,P,kappa)
    xm, cov = unscented_transform(xi, w)'''
    #test_radar()
    #test_sigma_plot()
    #test_julier_weights()
    #test_scaled_weights()

    #print('xi=\n',Xi)
    """
    xm, cov = unscented_transform(Xi, W)
    print(xm)
    print(cov)"""
#    sigma_points ([5,2],9*np.eye(2), 2)

