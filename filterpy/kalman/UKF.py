# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import scipy.linalg as linalg
import numpy as np


class UnscentedKalmanFilter(object):

    def __init__(self, dim_x, dim_z, kappa, dt):
        """docstring"""

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
        """docstring"""

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

    **Parameters**

    X An array of the means for each dimension in the problem space.
        Can be a scalar if 1D.
        examples: 1, [1,2], np.array([1,2])

    P : scalar, or2

    **Returns**

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

