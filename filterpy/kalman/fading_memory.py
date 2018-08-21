# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes


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


from __future__ import (absolute_import, division, unicode_literals)
from copy import deepcopy
from math import log, exp, sqrt
import sys
import warnings
import numpy as np
from numpy import dot, zeros, eye
import scipy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str

class FadingKalmanFilter(object):
    """
    Fading memory Kalman filter. This implements a linear Kalman filter with
    a fading memory effect controlled by `alpha`. This is obsolete. The
    class KalmanFilter now incorporates the `alpha` attribute, and should
    be used instead.

    You are responsible for setting the
    various state variables to reasonable values; the defaults below
    will not give you a functional filter.

    Parameters
    ----------

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

    Attributes
    ----------

    You will have to assign reasonable values to all of these before
    running the filter. All must have dtype of float

    x : ndarray (dim_x, 1), default = [0,0,0...0]
        state of the filter

    P : ndarray (dim_x, dim_x), default identity matrix
        covariance matrix

    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    z : ndarray
        Last measurement used in update(). Read only.

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

    y : numpy.array
        Residual of the update step. Read only.

    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.

    S :  numpy.array
        System uncertainty (P projected to measurement space). Read only.

    S :  numpy.array
        Inverse system uncertainty. Read only.

    log_likelihood : float
        log-likelihood of the last measurement. Read only.

    likelihood : float
        likelihood of last measurement. Read only.

        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.

    mahalanobis : float
        mahalanobis distance of the innovation. Read only.


    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """


    def __init__(self, alpha, dim_x, dim_z, dim_u=0):

        warnings.warn(
            "Use KalmanFilter class instead; it also provides the alpha attribute",
            DeprecationWarning)

        assert alpha >= 1
        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.alpha_sq = alpha**2
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))     # state
        self.P = eye(dim_x)            # uncertainty covariance
        self.Q = eye(dim_x)            # process uncertainty
        self.B = 0.                    # control transition matrix
        self.F = np.eye(dim_x)         # state transition matrix
        self.H = zeros((dim_z, dim_x)) # Measurement function
        self.R = eye(dim_z)            # state uncertainty
        self.z = np.array([[None]*dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))   # system uncertainty (measurement space)
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self.I = np.eye(dim_x)

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update(self, z, R=None):
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
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(self.H, self.x)

        PHT = dot(self.P, self.H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(self.H, PHT) + R
        self.SI = linalg.inv(self.S)

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = PHT.dot(self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - dot(self.K, self.H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

            # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def predict(self, u=0):
        """ Predict next position.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        # x = Fx + Bu
        self.x = dot(self.F, self.x) + dot(self.B, u)

        # P = FPF' + Q
        self.P = self.alpha_sq * dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

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

        n = np.size(zs, 0)
        if Rs is None:
            Rs = [None] * n

        #pylint: disable=bad-whitespace

        # mean estimates from Kalman Filter
        means   = zeros((n, self.dim_x, 1))
        means_p = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances   = zeros((n, self.dim_x, self.dim_x))
        covariances_p = zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, r) in enumerate(zip(zs, Rs)):
                self.update(z, r)
                means[i, :]          = self.x
                covariances[i, :, :] = self.P

                self.predict()
                means_p[i, :]          = self.x
                covariances_p[i, :, :] = self.P
        else:
            for i, (z, r) in enumerate(zip(zs, Rs)):
                self.predict()
                means_p[i, :]          = self.x
                covariances_p[i, :, :] = self.P

                self.update(z, r)
                means[i, :]          = self.x
                covariances[i, :, :] = self.P

        return (means, covariances, means_p, covariances_p)


    def get_prediction(self, u=0):
        """ Predicts the next state of the filter and returns it. Does not
        alter the state of the filter.

        Parameters
        ----------

        u : np.array
            optional control input

        Returns
        -------

        (x, P)
            State vector and covariance array of the prediction.
        """

        x = dot(self.F, self.x) + dot(self.B, u)
        P = self.alpha_sq * dot(self.F, self.P).dot(self.F.T) + self.Q
        return (x, P)


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
    def alpha(self):
        """ scaling factor for fading memory"""

        return sqrt(self.alpha_sq)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    def __repr__(self):
        return '\n'.join([
            'FadingKalmanFilter object',
            pretty_str('dim_x', self.x),
            pretty_str('dim_z', self.x),
            pretty_str('dim_u', self.dim_u),
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('F', self.F),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('H', self.H),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('S', self.S),
            pretty_str('B', self.B),
            pretty_str('likelihood', self.likelihood),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('mahalanobis', self.mahalanobis),
            pretty_str('alpha', self.alpha)
            ])
