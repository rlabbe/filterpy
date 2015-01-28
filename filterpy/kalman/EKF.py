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
import scipy.linalg as linalg
from numpy import dot, zeros, eye
from filterpy.common import setter, setter_1d, setter_scalar, dot3


class ExtendedKalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        """ Extended Kalman filter. You are responsible for setting the
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
        """

        self.dim_x = dim_x
        self.dim_z = dim_z

        self._x = zeros((dim_x,1)) # state
        self._P = eye(dim_x)       # uncertainty covariance
        self._B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._R = eye(dim_z)       # state uncertainty
        self._Q = eye(dim_x)       # process uncertainty
        self._y = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


    def predict_update(self, z, HJacobian, Hx, u=0):
        """ Performs the predict/update innovation of the extended Kalman
        filter.

        **Parameters**

        z : np.array
            measurement for this step.
            If `None`, only predict step is perfomed.

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.


        Hx : function
            function which takes a state variable and returns the measurement
            that would correspond to that state.

        u : np.array or scalar
            optional control vector input to the filter.
        """
        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        F = self._F
        B = self._B
        P = self._P
        Q = self._Q
        R = self._R
        x = self._x

        H = HJacobian(x)

        # predict step
        x = dot(F, x) + dot(B, u)
        P = dot3(F, P, F.T) + Q

        # update step
        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))

        self._x = x + dot(K, (z - Hx(x)))

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def update(self, z, HJacobian, Hx, R=None):
        """ Performs the update innovation of the extended Kalman filter.

        **Parameters**

        z : np.array
            measurement for this step.
            If `None`, only predict step is perfomed.

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.


        Hx : function
            function which takes a state variable and returns the measurement
            that would correspond to that state.
        """

        P = self._P
        if R is None:
            R = self._R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        x = self._x

        H = HJacobian(x)

        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))

        y = z - Hx(x)
        self._x = x + dot(K, y)

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def predict_x(self, u=0):
        """ predicts the next state of X. If you need to
        compute the next state yourself, override this function. You would
        need to do this, for example, if the usual Taylor expansion to
        generate F is not providing accurate results for you. """

        self._x = dot(self._F, self._x) + dot(self._B, u)


    def predict(self, u=0):
        """ Predict next position.

        **Parameters**

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        self.predict_x(u)
        self._P = dot3(self._F, self._P, self._F.T) + self._Q


    @property
    def Q(self):
        """ Process uncertainty"""
        return self._Q


    @Q.setter
    def Q(self, value):
        self._Q = setter_scalar(value, self.dim_x)


    @property
    def P(self):
        """ covariance matrix"""
        return self._P


    @P.setter
    def P(self, value):
        self._P = setter_scalar(value, self.dim_x)


    @property
    def R(self):
        """ measurement uncertainty"""
        return self._R


    @R.setter
    def R(self, value):
        self._R = setter_scalar(value, self.dim_z)


    @property
    def F(self):
        return self._F


    @F.setter
    def F(self, value):
        self._F = setter(value, self.dim_x, self.dim_x)


    @property
    def B(self):
        return self._B


    @B.setter
    def B(self, value):
        """ control transition matrix"""
        self._B = setter(value, self.dim_x, self.dim_u)


    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = setter_1d(value, self.dim_x)

    @property
    def K(self):
        """ Kalman gain """
        return self._K

    @property
    def y(self):
        """ measurement residual (innovation) """
        return self._y

    @property
    def S(self):
        """ system uncertainty in measurement space """
        return self._S


