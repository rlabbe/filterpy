# -*- coding: utf-8 -*-

"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import matplotlib.pyplot as plt

import numpy.random as random
from numpy.random import randn
import math
import numpy as np
import stats
from filterpy.kalman import UKF



def plot_sigma_test():
    """ Test to make sure sigma's correctly mirror the shape and orientation
    of the covariance array."""

    x = np.array([[1,2]])
    P = np.array([[2, 1.2],
                  [1.2, 2]])
    kappa = .1

    # if kappa is larger, than points shoudld be closer together

    Xi, W = sigma_points (x, P, kappa)
    for i in range(Xi.shape[0]):
        plt.scatter((Xi[i,0]-x[0,0])*W[i]+x[0,0],
                    (Xi[i,1]-x[0,1])*W[i]+x[0,1], color='blue')

    Xi, W = sigma_points (x, P, kappa*1000)
    for i in range(Xi.shape[0]):
        plt.scatter((Xi[i,0]-x[0,0])*W[i]+x[0,0],
                    (Xi[i,1]-x[0,1])*W[i]+x[0,1], color='green')

    stats.plot_covariance_ellipse([1,2],P)


def sigma_points_1D_tests():
    """ tests passing 1D data into sigma_points"""
    Xi, W = sigma_points (5,9,2)
    xm, cov = unscented_transform(Xi, W)

    assert Xi.shape == (3,1)
    assert len(W) == 3

    print('Xi=',Xi)
    print('W=',W)

    print('xm',xm)
    print('cov',cov)




class RadarSim(object):
    def __init__(self, dt):
        self.x = 0
        self.dt = dt

    def get_range(self):

        vel = 100 * 5*randn()
        alt = 1000 + 10*randn()
        self.x += vel*self.dt

        v = self.x * 0.05*randn()
        rng = (self.x**2 + alt**2)**.5 + v
        return rng


def GetRadar(dt):
    """ Simulate radar range to object at 1K altidue and moving at 100m/s.
    Adds about 5% measurement noise. Returns slant range to the object.
    Call once for each new measurement at dt time from last call.
    """

    if not hasattr (GetRadar, "posp"):
        GetRadar.posp = 0

    vel = 100  + 5 * randn()
    alt = 1000 + 10 * randn()
    pos = GetRadar.posp + vel*dt

    v = 0 + pos* 0.05*randn()
    range = math.sqrt (pos**2 + alt**2) + v
    GetRadar.posp = pos

    return range


def test_radar():
    def fx(x, dt):
        A = np.eye(3) + dt * np.array ([[0, 1, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]])
        return A.dot(x)

    def hx(x):
        return np.sqrt (x[0]**2 + x[2]**2)

    dt = 0.05
    kf = UKF(3,1,0,dt)
    kf.Q *= 0.01
    kf.R = 100
    kf.X = np.array([0., 90., 1100.])
    kf.P *= 100.
    radar = RadarSim(dt)

    t = np.arange(0,20+dt, dt)

    n = len(t)

    xs = np.zeros((n,3))

    random.seed(200)
    rs = []
    #xs = []
    for i in range(len(t)):
        #r = radar.get_range()
        r = GetRadar(dt)
        kf.update(r, fx, hx)

        xs[i,:] = kf.X
        rs.append(r)

    print(xs[:,0].shape)

    plt.subplot(311)
    plt.plot(t, xs[:,0])
    plt.subplot(312)
    plt.plot(t, xs[:,1])
    plt.subplot(313)

    plt.plot(t, xs[:,2])


if __name__ == "__main__":

    '''test_1D_sigma_points()
    #plot_sigma_test ()

    x = np.array([[1,2]])
    P = np.array([[2, 1.2],
                  [1.2, 2]])
    kappa = .1

    xi,w = sigma_points (x,P,kappa)
    xm, cov = unscented_transform(xi, w)'''
    test_radar()



    #print('xi=\n',Xi)
    """
    xm, cov = unscented_transform(Xi, W)
    print(xm)
    print(cov)"""
#    sigma_points ([5,2],9*np.eye(2), 2)

