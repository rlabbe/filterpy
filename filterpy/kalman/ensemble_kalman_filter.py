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
import scipy.linalg as linalg
from scipy.linalg import inv
from numpy import dot, zeros, eye, outer
from numpy.random import multivariate_normal
from filterpy.common import setter, setter_scalar, dot3



class EnsembleKalmanFilter(object):

    def __init__(self, dim_x, dim_z, dt, N, hx, fx):
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

        assert dim_x > 0
        assert dim_z > 0

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.N = N
        self.hx = hx
        self.fx = fx

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.R = eye(dim_z)       # state uncertainty

        self.sigmas = None
        self.sigmas_f = zeros((N, dim_x))

    def initialize(self, x, P):
        if x.ndim == 2:
            m = x[:,0]
        else:
            assert x.ndim == 1
            m = x
        self.sigmas = multivariate_normal(mean=m, cov=P, size=self.N)



    def update(self, z, R=None, H=None):
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
        for i, sigma in enumerate(self.sigmas):
            sigmas_h[i] = self.hx(sigma)

        z_k = np.mean(sigmas_h, axis=0)


        P_zz = 0
        for sigma in sigmas_h:
            P_zz += outer(sigma, sigma)

        P_zz = (P_zz / N) - outer(z_k, z_k) + R

        P_xz = 0
        for i in range(N):
            P_xz += outer(self.sigmas[i] - self.x, sigmas_h[i] - z_k)
        P_xz /= N

        K = dot(P_xz, inv(P_zz))

        e_r = multivariate_normal([0]*dim_z, R, N)

        for i in range(N):
            self.sigmas[i] += dot(K[i], z + e_r[i] - z_k)

        self.x = np.mean(self.sigmas_f, axis=0)
        self.P = self.P - dot3(K, P_zz, K.T)


    def predict(self, u=0):
        """ Predict next position.

        **Parameters**

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        x = 0
        P = 0
        mean = [0]*self.dim_x
        for i, s in enumerate(self.sigmas):
            sigma = self.fx(s, self.dt)
            e = multivariate_normal(mean, self.Q)
            x += sigma + e
            P += outer(sigma, sigma)
            self.sigmas_f[i] = sigma+e

        self.xp = x / self.N
        self.Pp = P / self.N - outer (self.x, self.x)


def f1():
    su = 0
    for i in range(len(sigmas)):
        su += np.dot (sigmas[i], 3)

def f2():
    su = 0
    for i,s in enumerate(sigmas):
        su += np.dot (s, 3)

