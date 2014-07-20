# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

This is licensed under an MIT license. See the readme.MD file
for more information.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)



import numpy.linalg as linalg
import numpy.random as random
from numpy.random import randn
import math
import numpy as np
import stats


class UKF(object):

    def __init__(self, dim_x, dim_z, kappa, dt):

        self.Q = np.eye(dim_x)
        self.R = 100
        self.X = np.zeros (dim_x)
        self.P = np.eye(dim_x)
        self.n = dim_x
        self.m = dim_z
        self.kappa = kappa
        self.dt = dt

        self.fXi = np.zeros((2*self.n+1,self.n))
        self.hXi = np.zeros((2*self.n+1,self.m))
        self.Pxz = np.zeros((self.m,self.n))


    def update(self, z, fx, hx):

        num_sigmas = 2*self.n + 1

        Xi, W = sigma_points (self.X, self.P, self.kappa)

        for i in range(num_sigmas):
            self.fXi[i] = fx(Xi[i], self.dt)


        xp, Pp = unscented_transform(self.fXi, W, self.Q)

        for i in range(num_sigmas):
            self.hXi[i] = hx(self.fXi[i])

        zp, Pz = unscented_transform(self.hXi, W, self.R)


        self.Pxz = np.zeros((self.m,self.n))
        for i in range(num_sigmas):
            self.Pxz += W[i] * (self.fXi[i] - xp) * (self.hXi[i] - zp).T

        K = self.Pxz.T.dot(linalg.inv(Pz))

        self.X = xp + K.dot(z-zp)
        self.P = Pp - K.dot(Pz).dot(K.T)



def sigma_points (X, P, kappa):
    """ Computes the sigma points and weights for an unscented Kalman filter
    given the mean and covariance of the filter.
    kappa is an arbitrary constant
    constant. Returns tuple of the sigma points and weights.

    Works with both scalar and array inputs:
    sigma_points (5, 9, 2) # mean 5, covariance 9
    sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

    Parameters
    ----------
    X An array of the means for each dimension in the problem space.
        Can be a scalar if 1D.
        examples: 1, [1,2], np.array([1,2])

    P : scalar, or

    Returns
    -------
    sigmas : np.array, of size (n, 2n+1)
        Two dimensional array of sigma points. Each column contains all of
        the sigmas for one dimension in the problem space.

        Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}

    weights : 1D np.array, of size (2n+1)
    """

    if np.isscalar(X):
        X = np.array([X])

    if  np.isscalar(P):
        P = np.array([[P]])

    """ Xi - sigma points
        W  - weights
    """

    n = np.size(X)  # dimension of problem

    W = np.full((2*n+1,1), .5 / (n+kappa))
    Xi = np.zeros((2*n+1, n))

    # handle values for the mean separately as special case
    Xi[0] = X
    W[0] = kappa / (n+kappa)

    # implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
    # Take transpose so we can access with U[i]
    U = linalg.cholesky((n+kappa)*P).T

    for k in range (n):
        Xi[k+1]   = X + U[k]
        Xi[n+k+1] = X - U[k]

    return (Xi, W)




def unscented_transform (Xi, W, NoiseCov=None):
    """ computes the unscented transform of a set of signma points and weights.
    returns the mean and covariance in a tuple
    """
    kmax,n = Xi.shape

    X = np.sum (Xi*W, axis=0)
    P = np.zeros((n,n))

    for k in range (kmax):
        s = (Xi[k]-X)[np.newaxis] # needs to be 2D to perform transform
        P += W[k,0]*s*s.T

    if NoiseCov is not None:
        P += NoiseCov

    return (X, P)


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


def test_1D_sigma_points():
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
        from numpy.random import randn

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
    import matplotlib.pyplot as plt


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

