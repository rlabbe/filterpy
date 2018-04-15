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


from __future__ import (absolute_import, division)
import math
import numpy as np
from numpy import dot, zeros, eye
from scipy.linalg import inv
import sys
from filterpy.stats import logpdf


class InformationFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0, compute_log_likelihood=True):
        """ Create a Information filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        Parameters
        ----------

        dim_x : int
            Number of state variables for the  filter. For example, if you
            are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        dim_u : int (optional)
            size of the control input, if it is being used.
            Default value of 0 indicates it is not used.

        self.compute_log_likelihood = compute_log_likelihood
        self.log_likelihood = math.log(sys.float_info.min)

        Examples
        --------

        See my book Kalman and Bayesian Filters in Python
        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
        """

        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1)) # state
        self.P_inv = eye(dim_x)   # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._F_inv = 0            # state transition matrix
        self.H = 0                # Measurement function
        self.R_inv = eye(dim_z)   # state uncertainty

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = 0 # system uncertainty in measurement space

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)
        self._no_information = False

        self.compute_log_likelihood = compute_log_likelihood
        self.log_likelihood = math.log(sys.float_info.min)


    def update(self, z, R_inv=None):
        """
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        """

        if z is None:
            return

        if R_inv is None:
            R_inv = self.R_inv
        elif np.isscalar(R_inv):
            R_inv = eye(self.dim_z) * R_inv

        # rename for readability and a tiny extra bit of speed
        H = self.H
        H_T = H.T
        P_inv = self.P_inv
        x = self.x

        if self._no_information:
            self.x = dot(P_inv, x) + dot(H_T, R_inv).dot(z)
            self.P_inv = P_inv + dot(H_T, R_inv).dot(H)
            self.log_likelihood = math.log(sys.float_info.min)

        else:
            # y = z - Hx
            # error (residual) between measurement and prediction
            self.y = z - dot(H, x)

            # S = HPH' + R
            # project system uncertainty into measurement space
            self.S = P_inv + dot(H_T, R_inv).dot(H)
            self.K = dot(inv(self.S), H_T).dot(R_inv)

            # x = x + Ky
            # predict new x with residual scaled by the kalman gain
            self.x = x + dot(self.K, self.y)
            self.P_inv = P_inv + dot(H_T, R_inv).dot(H)

            if self.compute_log_likelihood:
                self.log_likelihood = logpdf(x=self.y, cov=self.S)


    def predict(self, u=0):
        """ Predict next position.

        Parameters
        ----------

        u : ndarray
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        # x = Fx + Bu

        A = dot(self._F_inv.T, self.P_inv).dot(self._F_inv)
        try:
            AI = inv(A)
            invertable = True
            if self._no_information:
                try:
                    self.x = dot(inv(self.P_inv), self.x)
                except:
                    self.x = dot(0, self.x)
                self._no_information = False
        except:
            invertable = False
            self._no_information  = True

        if invertable:
            self.x = dot(self._F, self.x) + dot(self.B, u)
            self.P_inv = inv(AI + self.Q)
        else:
            I_PF = self._I - dot(self.P_inv,self._F_inv)
            FTI = inv(self._F.T)
            FTIX = dot(FTI, self.x)
            AQI = inv(A + self.Q)
            self.x = dot(FTI, dot(I_PF, AQI).dot(FTIX))


    def batch_filter(self, zs, Rs=None, update_first=False):
        """ Batch processes a sequences of measurements.

        Parameters
        ----------

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

        Returns
        -------

        means: np.array((n,dim_x,1))
            array of the state for each time step. Each entry is an np.array.
            In other words `means[k,:]` is the state at step `k`.

        covariance: np.array((n,dim_x,dim_x))
            array of the covariances for each time step. In other words
            `covariance[k,:,:]` is the covariance at step `k`.
        """

        raise "this is not implemented yet"

        ''' this is a copy of the code from kalman_filter, it has not been
        turned into the information filter yet. DO NOT USE.'''

        n = np.size(zs, 0)
        if Rs is None:
            Rs = [None] * n

        # mean estimates from Kalman Filter
        means = zeros((n, self.dim_x,1))

        # state covariances from Kalman Filter
        covariances = zeros((n,self.dim_x, self.dim_x))

        if update_first:
            for i,(z,r) in enumerate(zip(zs, Rs)):
                self.update(z, r)
                means[i,:] = self.x
                covariances[i,:,:] = self._P
                self.predict()
        else:
            for i,(z,r) in enumerate(zip(zs, Rs)):
                self.predict()
                self.update(z, r)

                means[i,:] = self.x
                covariances[i,:,:] = self._P

        return (means, covariances)


    @property
    def likelihood(self):
        """
        likelihood of last measurment.

        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.

        But really, this is a bad measure because of the scaling that is
        involved - try to use log-likelihood in your equations!"""

        lh = math.exp(self.log_likelihood)
        if lh == 0:
             lh = sys.float_info.min


    @property
    def F(self):
        """State Transition matrix"""
        return self._F


    @F.setter
    def F(self, value):
        self._F = value
        self._F_inv = inv(self._F)
