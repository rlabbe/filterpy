# -*- coding: utf-8 -*-
"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import absolute_import, division
import numpy as np
from numpy import dot, zeros, eye
from numpy.linalg import multi_dot
import scipy.linalg as linalg


class HInfinityFilter(object):

    def __init__(self, dim_x, dim_z, dim_u, gamma):
        """ Create an H-Infinity filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        Parameters
        ----------

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
            
        gamma : float
            
        """

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.gamma = gamma

        self.x = zeros((dim_x, 1)) # state

        self.B = 0                # control transition matrix
        self.F = 0                # state transition matrix
        self.H = 0                # Measurement function (matrix)
        
        # Uncertainty covariance. 
        # In the beginning should be small if we are highly confident
        # of our initial state estimate
        self.P = eye(dim_x)                 

        # the noise of measurement inversed 
        # the less V, the more we believe in measurement)
        self._V_inv = zeros((dim_z, dim_z)) 
        
        # noise of process 
        # (the less the noise, the more we believe in prediction)
        self.W = zeros((dim_x, dim_x))      
        
        # process uncertainty
        self.Q = eye(dim_x)                 

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # H-Infinity gain matrix
        self.residual = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


    def update(self, Z, R = None):
        """
        Add a new measurement (Z) to the H-Infinity filter. If Z is None, nothing
        is changed.

        Parameters
        ----------

        Z : np.array
            measurement for this update.
            
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self._V will be used.

        """

        if Z is None:
            return

        if R is not None:
            self.V = R
        
        # rename for readability and a tiny extra bit of speed
        I = self._I
        gamma = self.gamma
        Q = self.Q
        H = self.H
        P = self.P
        x = self.x
        V_inv = self._V_inv
        F = self.F
        W = self.W

        # common subexpression H.T * V^-1.
        HTVI = dot(H.T, V_inv)

        if np.isscalar(P):
            L = linalg.inv(I - gamma * Q * P + dot(HTVI, H) * P)
        else:
            L = linalg.inv(I - gamma * dot(Q, P) + multi_dot([HTVI, H, P]))

        # common subexpression P*L
        PL = dot(P, L)

        K = multi_dot([F, PL, HTVI])  

        self.residual = Z - dot(H, x)

        # x = x + Ky
        # predict new x with residual scaled by the H-Infinity gain
        self.x = self.x + dot(K, self.residual)
        self.P = multi_dot([F, PL, F.T]) + W

        # force P to be symmetric
        self.P = (self.P + self.P.T) / 2


    '''def update_safe(self, Z):
        """ same as update(), except we perform a check to ensure that the
        eigenvalues are < 1. An exception is thrown if not. """

        update(Z)
        evalue = linalg.eig(self.P)'
    '''


    def predict(self, u = 0):
        """ Predict next position.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        # x = Fx + Bu
        self.x = dot(self.F, self.x) + dot(self.B, u)


    def batch_filter(self, Zs, Rs = None, update_first = False):
        """ Batch processes a sequences of measurements.

        Parameters
        ----------

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

        Returns
        -------

        means: np.array((n,dim_x,1))
            array of the state for each time step. Each entry is an np.array.
            In other words `means[k,:]` is the state at step `k`.

        covariance: np.array((n,dim_x,dim_x))
            array of the covariances for each time step. In other words
            `covariance[k,:,:]` is the covariance at step `k`.
        """

        n = np.size(Zs, 0)
        if Rs is None:
            Rs = [None] * n

        # mean estimates from Kalman Filter
        means = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, r) in enumerate(zip(Zs, Rs)):
                self.update(z, r)
                means[i, :] = self.x
                covariances[i, :, :] = self.P
                self.predict()
        else:
            for i, (z, r) in enumerate(zip(Zs, Rs)):
                self.predict()
                self.update(z, r)

                means[i, :] = self.x
                covariances[i, :, :] = self.P

        return (means, covariances)


    def get_prediction(self, u=0):
        """ Predicts the next state of the filter and returns it. Does not
        alter the state of the filter.

        Parameters
        ----------

        u : np.array
            optional control input

        Returns
        -------

        x : numpy.ndarray
            State vecto of the prediction.
        """

        x = dot(self.F, self.x) + dot(self.B, u)
        return x


    def residual_of(self, z):
        """ returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return z - dot(self.H, self.x)


    def measurement_of_state(self, x):
        """ Helper function that converts a state into a measurement.

        Parameters
        ----------

        x : np.array
            kalman state vector

        Returns
        -------

        z : np.array
            measurement corresponding to the given state
        """
        return dot(self.H, x)


    @property
    def V(self):
        return self._V


    @V.setter
    def V(self, value):
        if np.isscalar(value):
            self._V = np.array([[value]], dtype=float)
        else:
            self._V = value
        self._V_inv = linalg.inv(self._V)

