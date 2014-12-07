# -*- coding: utf-8 -*-

"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from scipy.linalg import inv
from numpy import dot, zeros, eye, outer
from numpy.random import multivariate_normal


class EnsembleKalmanFilter(object):

    def __init__(self, x, P, dim_z, dt, N, hx, fx):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        **Parameters**

        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        dim_u : int (optional)
            size of the control input, if it is being used.
            Default value of 0 indicates it is not used.


        """

        assert dim_z > 0

        self.dim_x = len(x)
        self.dim_z = dim_z
        self.dt = dt
        self.N = N
        self.hx = hx
        self.fx = fx

        self.Q = eye(self.dim_x)       # process uncertainty
        self.R = eye(self.dim_z)       # state uncertainty
        self.mean = [0]*self.dim_x
        self.initialize(x, P)


    def initialize(self, x, P):
        """ Initializes the filter with the specified mean and
        covariance. Only need to call this if you are using the filter
        to filter more than one set of data; this is called by __init__

        **Parameters**

        x : np.array(dim_z)
            state mean

        P : np.array((dim_x, dim_x))
            covariance of the state
        """
        assert x.ndim == 1
        self.sigmas = multivariate_normal(mean=x, cov=P, size=self.N)

        self.x = x
        self.P = P


    def update(self, z, R=None):
        """
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.

        **Parameters**

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        """

        if z is None:
            return

        if R is None:
            R = self.R

        if np.isscalar(R):
            R = eye(self.dim_z) * R

        N = self.N

        dim_z = len(z)
        sigmas_h = zeros((N, dim_z))

        # transform sigma points into measurement space
        for i in range(N):
            sigmas_h[i] = self.hx(self.sigmas[i])

        z_k = np.sum(sigmas_h, axis=0) / (N)

        P_zz = 0
        for sigma in sigmas_h:
            s = sigma - z_k
            P_zz += outer(s, s)

        P_zz = (P_zz / (N)) + R# - outer(z_k, z_k) + R

        P_xz = 0
        for i in range(N):
            P_xz += outer(self.sigmas[i] - self.x, sigmas_h[i] - z_k)

        P_xz /= (N)

        K = dot(P_xz, inv(P_zz))

        e_r = multivariate_normal([0]*dim_z, R, N)

        y = z - z_k

        for i in range(N):
            self.sigmas[i] += dot(K, z + e_r[i] - sigmas_h[i])

        #self.sigmas += dot(K, z - z_k)

        self.x = np.sum(self.sigmas, axis=0) / (N)
        #self.P = self.P - dot3(K, P_zz, K.T)


    def predict(self):
        """ Predict next position. """

        x = 0

        e = multivariate_normal(self.mean, self.Q, self.N)
        for i, s in enumerate(self.sigmas):
            self.sigmas[i] = self.fx(s, self.dt)

        self.sigmas += e

        self.x = np.sum(self.sigmas, axis=0) / (self.N-1)


def f1():
    su = 0
    for i in range(len(sigmas)):
        su += np.dot (sigmas[i], 3)

def f2():
    su = 0
    for i,s in enumerate(sigmas):
        su += np.dot (s, 3)

