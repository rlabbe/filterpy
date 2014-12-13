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
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from scipy.linalg import inv
from numpy import dot, zeros, eye, outer
from numpy.random import multivariate_normal
from filterpy.common import dot3


class EnsembleKalmanFilter(object):
    """ This implements the ensemble Kalman filter (EnKF). The EnKF uses
    an ensemble of hundreds to thousands of state vectors that are randomly
    sampled around the estimate, and adds perturbations at each update and
    predict step. It is useful for extremely large systems such as found
    in hydrophysics. As such, this class is admittedly a toy as it is far
    too slow with large N.

    There are many versions of this sort of this filter. This formulation is
    due to Crassidis and Junkins [1]. It works with both linear and nonlinear
    systems.

    **References**

    - [1] John L Crassidis and John L. Junkins. "Optimal Estimation of
      Dynamic Systems. CRC Press, second edition. 2012. pp, 257-9.
    """

    def __init__(self, x, P, dim_z, dt, N, hx, fx):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        **Parameters**

        x : np.array(dim_z)
            state mean

        P : np.array((dim_x, dim_x))
            covariance of the state

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        dt : float
            time step in seconds

        N : int
            number of sigma points (ensembles). Must be greater than 1.

        hx : function hx(x)
            Measurement function. May be linear or nonlinear - converts state
            x into a measurement. Return must be an np.array of the same
            dimensionality as the measurement vector.

        fx : function fx(x, dt)
            State transition function. May be linear or nonlinear. Projects
            state x into the next time period. Returns the projected state x.

        **Example**

        .. code::

            def hx(x):
               return np.array([x[0]])

            F = np.array([[1., 1.],
                          [0., 1.]])
            def fx(x, dt):
                return np.dot(F, x)

            x = np.array([0., 1.])
            P = np.eye(2) * 100.
            dt = 0.1
            f = EnKF(x=x, P=P, dim_z=1, dt=dt, N=8,
                     hx=hx, fx=fx)

            std_noise = 3.
            f.R *= std_noise**2
            f.Q = Q_discrete_white_noise(2, dt, .01)

            while True:
                z = read_sensor()
                f.predict()
                f.update(np.asarray([z]))

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

        z_mean = np.mean(sigmas_h, axis=0)

        P_zz = 0
        for sigma in sigmas_h:
            s = sigma - z_mean
            P_zz += outer(s, s)
        P_zz = P_zz / (N-1) + R

        P_xz = 0
        for i in range(N):
            P_xz += outer(self.sigmas[i] - self.x, sigmas_h[i] - z_mean)
        P_xz /= N-1

        K = dot(P_xz, inv(P_zz))

        e_r = multivariate_normal([0]*dim_z, R, N)
        for i in range(N):
            self.sigmas[i] += dot(K, z + e_r[i] - sigmas_h[i])

        self.x = np.mean(self.sigmas, axis=0)
        self.P = self.P - dot3(K, P_zz, K.T)


    def predict(self):
        """ Predict next position. """

        N = self.N
        for i, s in enumerate(self.sigmas):
           self.sigmas[i] = self.fx(s, self.dt)

        e = multivariate_normal(self.mean, self.Q, N)
        self.sigmas += e
        #self.x = np.mean(self.sigmas , axis=0)

        P = 0
        for s in self.sigmas:
            sx = s - self.x
            P += outer(sx, sx)

        self.P = P / (N-1)
