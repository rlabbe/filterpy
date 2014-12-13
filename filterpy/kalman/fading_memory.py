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
from numpy import dot, zeros, eye, asarray
from filterpy.common import setter, setter_scalar, dot3



class FadingKalmanFilter(object):

    def __init__(self, alpha, dim_x, dim_z, dim_u=0):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        **Parameters**

        alpha : float, >= 1
            alpha controls how much you want the filter to forget past
            measurements. alpha==1 yields identical performance to the
            Kalman filter. A typical application might use 1.01

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


        **Instance Variables**

        You will have to assign reasonable values to all of these before
        running the filter. All must have dtype of float

        X : ndarray (dim_x, 1), default = [0,0,0...0]
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
        """

        assert alpha >= 1
        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0


        self.alpha_sq = alpha**2
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self._X = zeros((dim_x,1)) # state
        self._P = eye(dim_x)       # uncertainty covariance
        self._Q = eye(dim_x)       # process uncertainty
        self._B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._H = 0                # Measurement function
        self._R = eye(dim_z)       # state uncertainty

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self._K = 0 # kalman gain
        self._y = zeros((dim_z, 1))
        self._S = 0 # system uncertainty in measurement space

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


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
            R = self._R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        H = self._H
        P = self._P
        x = self._X

        # y = z - Hx
        # error (residual) between measurement and prediction
        self._y = z - dot(H, x)

        # S = HPH' + R
        # project system uncertainty into measurement space
        S = dot3(H, P, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        K = dot3(P, H.T, linalg.inv(S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self._X = x + dot(K, self._y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)

        self._S = S
        self._K = K


    def predict(self, u=0):
        """ Predict next position.

        **Parameters**

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        # x = Fx + Bu
        self._X = dot(self._F, self._X) + dot(self._B, u)

        # P = FPF' + Q
        self._P = self.alpha_sq*dot3(self._F, self._P, self._F.T) + self._Q


    def batch_filter(self, zs, Rs=None, update_first=False):
        """ Batch processes a sequences of measurements.

        **Parameters**

        zs : list-like
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
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance: np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        means_predictions: np.array((n,dim_x,1))
            array of the state for each time step after the predictions. Each
            entry is an np.array. In other words `means[k,:]` is the state at
            step `k`.

        covariance_predictions: np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the prediction.
            In other words `covariance[k,:,:]` is the covariance at step `k`.
        """

        n = np.size(zs,0)
        if Rs is None:
            Rs = [None]*n

        # mean estimates from Kalman Filter
        means   = zeros((n,self.dim_x,1))
        means_p = zeros((n,self.dim_x,1))

        # state covariances from Kalman Filter
        covariances   = zeros((n,self.dim_x,self.dim_x))
        covariances_p = zeros((n,self.dim_x,self.dim_x))

        if update_first:
            for i,(z,r) in enumerate(zip(zs,Rs)):
                self.update(z,r)
                means[i,:]         = self._X
                covariances[i,:,:] = self._P

                self.predict()
                means_p[i,:]         = self._X
                covariances_p[i,:,:] = self._P
        else:
            for i,(z,r) in enumerate(zip(zs,Rs)):
                self.predict()
                means_p[i,:]         = self._X
                covariances_p[i,:,:] = self._P

                self.update(z,r)
                means[i,:]         = self._X
                covariances[i,:,:] = self._P

        return (means, covariances, means_p, covariances_p)


    def get_prediction(self, u=0):
        """ Predicts the next state of the filter and returns it. Does not
        alter the state of the filter.

        **Parameters**

        u : np.array
            optional control input

        **Returns**

        (x, P)
            State vector and covariance array of the prediction.
        """

        x = dot(self._F, self._X) + dot(self._B, u)
        P = self.alpha_sq*dot3(self._F, self._P, self._F.T) + self.Q
        return (x, P)


    def residual_of(self, z):
        """ returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return z - dot(self._H, self._X)


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
    def X(self):
        """ filter state vector."""
        return self._X


    @X.setter
    def X(self, value):
        self._X = setter(value, self.dim_x, 1)


    @property
    def x(self):
        assert False

    @x.setter
    def x(self, value):
        assert False

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
        """ system uncertainy in measurement space """
        return self._S
