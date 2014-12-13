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
from scipy.linalg import cholesky, qr, pinv
from numpy import dot, zeros, eye
from filterpy.common import setter, setter_scalar, dot3





class SquareRootKalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        """ Create a Kalman filter which uses a square root implementation.
        This uses the square root of the state covariance matrix, which doubles
        the numerical precision of the filter, Therebuy reducing the effect
        of round off errors.

        It is likely that you do not need to use this algorithm; we understand
        divergence issues very well now. However, if you expect the covariance
        matrix P to vary by 20 or more orders of magnitude then perhaps this
        will be useful to you, as the square root will vary by 10 orders
        of magnitude. From my point of view this is merely a 'reference'
        algorithm; I have not used this code in real world software. Brown[1]
        has a useful discussion of when you might need to use the square
        root form of this algorithm.

        You are responsible for setting the various state variables to
        reasonable values; the defaults below will not give you a functional
        filter.

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


        **Instance Variables:**

        You will have to assign reasonable values to all of these before
        running the filter. All must have dtype of float

        x : ndarray (dim_x, 1), default = [0,0,0...0]
            state of the filter

        P : ndarray (dim_x, dim_x), default identity matrix
            covariance matrix

        Q : ndarray (dim_x, dim_x), default identity matrix
            Process uncertainty matrix

        R : ndarray (dim_z, dim_z), default identity matrix
            measurement uncertainty

        H : ndarray (dim_z, dim_x)
            measurement function

        F : ndarray (dim_x, dim_x)
            state transistion matrix

        B : ndarray (dim_x, dim_u), default 0
            control transition matrix


        **References**

        [1] Robert Grover Brown. Introduction to Random Signals and Applied
            Kalman Filtering. Wiley and sons, 2012.
        """

        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self._x = zeros((dim_x,1)) # state
        self._P = eye(dim_x)      # uncertainty covariance
        self._P1_2 = eye(dim_x)      # sqrt uncertainty covariance
        self._Q = eye(dim_x)      # sqrt process uncertainty
        self._Q1_2 = eye(dim_x)      # sqrt process uncertainty
        self._B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._H = 0                # Measurement function
        self._R1_2 = eye(dim_z)      # sqrt state uncertainty

        # Residual is computed during the innovation (update) step. We
        # save it so that in case you want to inspect it for various
        # purposes
        self._y = zeros((dim_z, 1))

        # identity matrix.
        self._I = np.eye(dim_x)

        self._M = np.zeros((dim_z + dim_x, dim_z + dim_x))


    def update(self, z, R2=None):
        """
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.

        **Parameters**

        z : np.array
            measurement for this update.

        R2 : np.array, scalar, or None
            Sqrt of meaaurement noize. Optionally provide to override the
            measurement noise for this one call, otherwise  self.R2 will
            be used.
        """

        if z is None:
            return

        if R2 is None:
            R2 = self._R1_2
        elif np.isscalar(R2):
            R2 = eye(self.dim_z) * R2

        # rename for convienance
        dim_z = self.dim_z
        M = self._M

        M[0:dim_z, 0:dim_z] = R2.T
        M[dim_z:, 0:dim_z] = dot(self._H, self._P1_2).T
        M[dim_z:, dim_z:] = self._P1_2.T

        _, S = qr(M)
        self._K = S[0:dim_z,  dim_z:].T
        N = S[0:dim_z, 0:dim_z].T

        # y = z - Hx
        # error (residual) between measurement and prediction
        self._y = z - dot(self._H, self._x)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self._x += dot3(self._K, pinv(N), self._y)
        self._P1_2 = S[dim_z:, dim_z:].T



    def predict(self, u=0):
        """ Predict next position.

        **Parameters**

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        # x = Fx + Bu
        self._x = dot(self._F, self.x) + dot(self._B, u)

        # P = FPF' + Q
        T,P2 = qr(np.hstack([dot(self._F, self._P1_2), self._Q1_2]).T)
        self._P1_2 = P2[:self.dim_x, :self.dim_x].T


    def residual_of(self, z):
        """ returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """

        return z - dot(self._H, self._x)


    def measurement_of_state(self, x):
        """ Helper function that converts a state into a measurement.

        **Parameters**

        x : np.array
            kalman state vector

        **Returns**

        z : np.array
            measurement corresponding to the given state
        """
        return dot(self._H, x)


    @property
    def Q(self):
        """ Process uncertainty"""
        return dot(self._Q1_2.T, self._Q1_2)


    @property
    def Q1_2(self):
        """ Sqrt Process uncertainty"""
        return self._Q1_2


    @Q.setter
    def Q(self, value):
        self._Q = setter_scalar(value, self.dim_x)
        self._Q1_2 = cholesky (self._Q, lower=True)

    @property
    def P(self):
        """ covariance matrix"""
        return dot(self._P1_2.T, self._P1_2)


    @property
    def P1_2(self):
        """ sqrt of covariance matrix"""
        return self._P1_2


    @P.setter
    def P(self, value):
        self._P = setter_scalar(value, self.dim_x)
        self._P1_2 = cholesky(self._P, lower=True)


    @property
    def R(self):
        """ measurement uncertainty"""
        return dot(self._R1_2.T, self._R1_2)

    @property
    def R1_2(self):
        """ sqrt of measurement uncertainty"""
        return self._R1_2


    @R.setter
    def R(self, value):
        self._R = setter_scalar(value, self.dim_z)
        self._R1_2 = cholesky (self._R, lower=True)

    @property
    def H(self):
        """ Measurement function"""
        return self._H


    @H.setter
    def H(self, value):
        self._H = setter(value, self.dim_z, self.dim_x)


    @property
    def F(self):
        """ state transition matrix"""
        return self._F


    @F.setter
    def F(self, value):
        self._F = setter(value, self.dim_x, self.dim_x)

    @property
    def B(self):
        """ control transition matrix"""
        return self._B


    @B.setter
    def B(self, value):
        """ control transition matrix"""
        self._B = setter (value, self.dim_x, self.dim_u)


    @property
    def x(self):
        """ filter state vector."""
        return self._x


    @x.setter
    def x(self, value):
        self._x = setter(value, self.dim_x, 1)

    @property
    def K(self):
        """ Kalman gain """
        return self._K

    @property
    def y(self):
        """ measurement residual (innovation) """
        return self._y
