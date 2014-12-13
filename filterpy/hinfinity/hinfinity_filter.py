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
from filterpy.common import setter, setter_scalar, dot3


class HInfinityFilter(object):

    def __init__(self, dim_x, dim_z, dim_u, gamma):
        """ Create an H-Infinity filter. You are responsible for setting the
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

        dim_u : int
            Number of control inputs for the Gu part of the prediction step.
        """

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.gamma = gamma

        self.x = zeros((dim_x,1)) # state

        self._G = 0                # control transistion matrx
        self._F = 0                # state transition matrix
        self._H = 0                # Measurement function

        self._P = eye(dim_x)       # uncertainty covariance
        self._V_inv = zeros((dim_z, dim_z))
        self._W = zeros((dim_x, dim_x))
        self._Q = eye(dim_x)       # process uncertainty

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.residual = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


    def update(self, Z):
        """
        Add a new measurement (Z) to the kalman filter. If Z is None, nothing
        is changed.

        **Parameters**

        Z : np.array
            measurement for this update.
        """

        if Z is None:
            return

        # rename for readability and a tiny extra bit of speed
        I = self._I
        gamma = self.gamma
        Q = self._Q
        H = self._H
        P = self._P
        x = self._x
        V_inv = self._V_inv
        F = self._F
        W = self._W

        # common subexpression H.T * V^-1
        HTVI = dot(H.T, V_inv)

        L = linalg.inv(I - gamma*dot(Q, P) + dot3(HTVI, H, P))

        #common subexpression P*L
        PL = dot(P,L)

        K = dot3(F, PL, HTVI)

        self.residual = Z - dot(H, x)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self._x = self._x + dot(K, self.residual)
        self._P = dot3(F, PL, F.T) + W

        # force P to be symmetric
        self._P = (self._P + self._P.T) / 2


    '''def update_safe(self, Z):
        """ same as update(), except we perform a check to ensure that the
        eigenvalues are < 1. An exception is thrown if not. """

        update(Z)
        evalue = linalg.eig(self.P)'
    '''


    def predict(self, u=0):
        """ Predict next position.

        **Parameters**

        u : np.array
            Optional control vector. If non-zero, it is multiplied by G
            to create the control input into the system.
        """

        # x = Fx + Gu
        self._x = dot(self._F, self._x) + dot(self._G, u)


    def batch_filter(self, Zs, Rs=None, update_first=False):
        """ Batch processes a sequences of measurements.

        **Parameters**

        Zs : list-like
            list of measurements at each time step `self.dt` Missing
            measurements must be represented by 'None'.

        Rs : list-like, optional
            optional list of values to use for the measurement error
            covariance; a value of None in any position will cause the filter
            to use `self.R` for that time step.

        update_first : bool, optional,
            controls whether the order of operations is update followed by
            predict, or predict followed by update. Default is predict->update.

        **Returns**

        means: np.array((n,dim_x,1))
            array of the state for each time step. Each entry is an np.array.
            In other words `means[k,:]` is the state at step `k`.

        covariance: np.array((n,dim_x,dim_x))
            array of the covariances for each time step. In other words
            `covariance[k,:,:]` is the covariance at step `k`.
        """

        n = np.size(Zs,0)
        if Rs is None:
            Rs = [None]*n

        # mean estimates from Kalman Filter
        means = zeros((n,self.dim_x,1))

        # state covariances from Kalman Filter
        covariances = zeros((n,self.dim_x,self.dim_x))

        if update_first:
            for i,(z,r) in enumerate(zip(Zs,Rs)):
                self.update(z,r)
                means[i,:] = self.x
                covariances[i,:,:] = self.P
                self.predict()
        else:
            for i,(z,r) in enumerate(zip(Zs,Rs)):
                self.predict()
                self.update(z,r)

                means[i,:] = self.x
                covariances[i,:,:] = self.P

        return (means, covariances)


    def get_prediction(self, u=0):
        """ Predicts the next state of the filter and returns it. Does not
        alter the state of the filter.

        **Parameters**

        u : np.array
            optional control input

        **Returns**

        x : numpy.ndarray
            State vecto of the prediction.
        """

        x = dot(self._F, self._x) + dot(self._G, u)
        return x


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
    def x(self):
        """ state vector property"""
        return self._x


    @x.setter
    def x(self, value):
        self._x = setter(value, self.dim_x, 1)


    @property
    def G(self):
        return self._G


    @G.setter
    def G(self, value):
        self._G = setter(self.dim_x, 1)


    @property
    def P(self):
        """ covariance matrix property"""
        return self._P


    @P.setter
    def P(self, value):
        self._P = setter_scalar(value, self.dim_x)


    @property
    def F(self):
        return self._F


    @F.setter
    def F(self, value):
        self._F = setter(value, self.dim_x, self.dim_x)


    @property
    def G(self):
        return self._G


    @G.setter
    def G(self, value):
        self._G = setter(value, self.dim_x, self.dim_u)


    @property
    def H(self):
        return self._H


    @H.setter
    def H(self, value):
        self._H = setter(value, self.dim_z, self.dim_x)


    @property
    def V(self):
        return self._V


    @V.setter
    def V(self, value):
        self._V = setter_scalar(value, self.dim_z)
        self._V_inv = linalg.inv(self.V)


    @property
    def W(self):
        return self._W


    @W.setter
    def W(self, value):
        self._W = setter_scalar(value, self.dim_x)


    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = setter_scalar(value, self.dim_x)
