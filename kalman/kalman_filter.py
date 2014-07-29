# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

This is licensed under an MIT license. See the readme.MD file
for more information.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import scipy.linalg as linalg
from numpy import dot, zeros, eye


class KalmanFilter:

    def __init__(self, dim_x, dim_z):
        """ Create a Kalman filter. You are responsible for setting the
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
        """

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.u = 0                # control input vector
        self.B = 0
        self.F = 0                # state transition matrix
        self.H = 0                # Measurement function
        self.R = eye(dim_z)       # state uncertainty
        self.K = 0                # kalman gain

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


    def update(self, Z, R=None):
        """
        Add a new measurement (Z) to the kalman filter. If Z is None, nothing
        is changed.

        Optionally provide R to override the measurement noise for this
        one call, otherwise  self.R will be used.

        self.residual, self.S, and self.K are stored in case you want to
        inspect these variables. Strictly speaking they are not part of the
        output of the Kalman filter, however, it is often useful to know
        what these values are in various scenarios.
        """

        if Z is None:
            return

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        H = self.H
        P = self.P
        x = self.x

        # y = Z - Hx
        # error (residual) between measurement and prediction
        self.residual = Z - dot(H, x)

        # S = HPH' + R
        # project system uncertainty into measurement space
        S = dot(dot(H, P), H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        K = dot(dot(P, H.T), linalg.inv(S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(K, self.residual)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(K, H)
        self.P = dot(I_KH, dot(P, I_KH.T)) + dot(K, dot(R, K.T))

        self.S = S
        self.K = K


    def predict(self, ekf_x=None):
        """ Predict next position. For a linear Kalman filter, leave
        `ekf_x` set to None. On the other hand, if you are implementing
        an Extended Kalman filter you will need to compute the nonlinear
        state transition yourself. There is no one way to do this, so you
        must compute the result based on the design of your filter. Then,
        pass the result in as the new state vector. self.x will be set to
        ekf_x.
        """

        if ekf_x is None:
            # x = Fx + Bu
            self.x = dot(self.F, self.x) + dot(self.B, self.u)
        else:
            self.x = ekf_x

        # P = FPF' + Q
        self.P = dot(dot(self.F, self.P), self.F.T) + self.Q


    def batch_filter(self, Zs, Rs=None, update_first=False):
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
