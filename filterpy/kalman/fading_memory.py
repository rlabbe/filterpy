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
import sys
import warnings
import math
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

    compute_log_likelihood : bool (default = True)
        Computes log likelihood by default, but this can be a slow
        computation, so if you never use it you can turn this computation
        off.

    Attributes
    ----------

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

    y : numpy.array
        Residual of the update step. Read only.

    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.

    S :  numpy.array
        Systen uncertaintly projected to measurement space. Read only.

    log_likelihood : float
        log-likelihood of the last measurement. Read only.


    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """


    def __init__(self, alpha, dim_x, dim_z, dim_u=0,
                 compute_log_likelihood=True):

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

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = 0 # system uncertainty in measurement space

        # identity matrix. Do not alter this.
        self.I = np.eye(dim_x)

        self.compute_log_likelihood = compute_log_likelihood
        self.log_likelihood = math.log(sys.float_info.min)


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

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = PHT.dot(linalg.inv(self.S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - dot(self.K, self.H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)


        if self.compute_log_likelihood:
            self.log_likelihood = logpdf(x=self.y, cov=self.S)


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

        return math.sqrt(self.alpha_sq)

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
        return lh

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
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('alpha', self.alpha),
            ])
