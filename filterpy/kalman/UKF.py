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

# pylint bug - warns about numpy functions which do in fact exist.
# pylint: disable=E1101


#I like aligning equal signs for readability of math
# pylint: disable=C0326

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy.linalg import inv, cholesky
import numpy as np
from numpy import asarray, eye, zeros, dot
from filterpy.common import dot3


class UnscentedKalmanFilter(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=C0103
    """ Implements the Unscented Kalman filter (UKF) as defined by Simon J.
    Julier and Jeffery K. Uhlmann [1]. Succintly, the UKF selects a set of
    sigma points and weights inside the covariance matrix of the filter's
    state. These points are transformed through the nonlinear process being
    filtered, and are rebuilt into a mean and covariance by computed the
    weighted mean and expected value of the transformed points. Read the paper;
    it is excellent. My book "Kalman and Bayesian Filters in Python" [2]
    explains the algorithm, develops this code, and provides examples of the
    filter in use.


    You will have to set the following attributes after constructing this
    object for the filter to perform properly.

    **Attributes**

    x : numpy.array(dim_x)
        state estimate vector

    P : numpy.array(dim_x, dim_x)
        covariance estimate matrix

    R : numpy.array(dim_z, dim_z)
        measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        process noise matrix


    You may read the following attributes.

    **Readable Attributes**

    xp : numpy.array(dim_x)
        predicted state (result of predict())

    Pp : numpy.array(dim_x, dim_x)
        predicted covariance matrix (result of predict())


    **References**

    .. [1] Julier, Simon J.; Uhlmann, Jeffrey "A New Extension of the Kalman
        Filter to Nonlinear Systems". Proc. SPIE 3068, Signal Processing,
        Sensor Fusion, and Target Recognition VI, 182 (July 28, 1997)

    .. [2] Labbe, Roger R. "Kalman and Bayesian Filters in Python"

        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, dim_x, dim_z, dt, hx, fx, kappa=0.):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        **Parameters**

        dim_x : int
            Number of state variables for the filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.


        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        dt : float
            Time between steps in seconds.

        hx : function(x)
            Measurement function. Converts state vector x into a measurement
            vector of shape (dim_z).

        fx : function(x,dt)
            function that returns the state x transformed by the
            state transistion function. dt is the time step in seconds.

        kappa : float, default=0.
            Scaling factor that can reduce high order errors. kappa=0 gives
            the standard unscented filter. According to [1], if you set
            kappa to 3-dim_x for a Gaussian x you will minimize the fourth
            order errors in x and P.

        **References**

        [1] S. Julier, J. Uhlmann, and H. Durrant-Whyte. "A new method for
            the nonlinear transformation of means and covariances in filters
            and estimators," IEEE Transactions on Automatic Control, 45(3),
            pp. 477-482 (March 2000).
        """

        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self._dt = dt
        self._num_sigmas = 2*dim_x + 1
        self.kappa = kappa
        self.hx = hx
        self.fx = fx

        # weights for the sigma points
        self.W = self.weights(dim_x, kappa)

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update
        self.sigmas_f = zeros((2*self._dim_x+1, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))


    def update(self, z, R=None, residual=np.subtract, UT=None):
        """ Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        **Parameters**

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work, but if for example you are using angles the default method
            of computing means and residuals will not work, and you will have
            to define how to compute it.
        """

        dim_z = len(z)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability
        sigmas_f = self.sigmas_f
        sigmas_h = zeros((self._num_sigmas, dim_z))


        if UT is None:
            UT = unscented_transform

        # transform sigma points into measurement space
        for i in range(self._num_sigmas):
            sigmas_h[i] = self.hx(sigmas_f[i])

        # mean and covariance of prediction passed through inscented transform
        zp, Pz = UT(sigmas_h, self.W, self.W, R)

        # compute cross variance of the state and the measurements
        Pxz = zeros((self._dim_x, dim_z))
        for i in range(self._num_sigmas):
            Pxz += self.W[i] * np.outer(sigmas_f[i] - self.x,
                                        residual(sigmas_h[i], zp))

        K = dot(Pxz, inv(Pz)) # Kalman gain

        y = residual(z, zp)

        self.x = self.x + dot(K, y)
        self.P = self.P - dot3(K, Pz, K.T)


    def predict(self):
        """ Performs the predict step of the UKF. On return, self.xp and
        self.Pp contain the predicted state (xp) and covariance (Pp). 'p'
        stands for prediction.

        Important: this MUST be called before update() is called for the first
        time.
        """

        # calculate sigma points for given mean and covariance
        sigmas = self.sigma_points(self.x, self.P, self.kappa)

        for i in range(self._num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], self._dt)

        self.x, self.P = unscented_transform(
                           self.sigmas_f, self.W, self.W, self.Q)


    @staticmethod
    def weights(n, kappa):
        """ Computes the weights for an unscented Kalman filter.  See
        __init__() for meaning of parameters.
        """

        k = .5 / (n+kappa)
        W = np.full(2*n+1, k)
        W[0] = kappa / (n+kappa)
        return W


    @staticmethod
    def sigma_points(x, P, kappa):
        """ Computes the sigma pointsfor an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        kappa is an arbitrary constant. Returns tuple of the sigma points
        and weights.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        **Parameters**

        X An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        kappa : float
            Scaling factor.

        **Returns**

        sigmas : np.array, of size (n, 2n+1)
            2D array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space. They
            are ordered as:

            .. math::
                sigmas[0]    = x \n
                sigmas[1..n] = x + [\sqrt{(n+\kappa)P}]_k \n
                sigmas[n+1..2n] = x - [\sqrt{(n+\kappa)P}]_k
        """

        if np.isscalar(x):
            x = asarray([x])
        n = np.size(x)  # dimension of problem

        if  np.isscalar(P):
            P = eye(n)*P

        sigmas = zeros((2*n+1, n))

        # implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        U = cholesky((n+kappa)*P).T
        #U = sqrtm((n+kappa)*P).T

        for k in range(n):
            sigmas[k+1]   = x + U[k]
            sigmas[n+k+1] = x - U[k]

        # handle value for the mean separately as special case
        sigmas[0] = x

        return sigmas


def unscented_transform(Sigmas, Wm, Wc, noise_cov):
    """ Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.
    """

    kmax, n = Sigmas.shape

    # new mean is just the sum of the sigmas * weight
    x = dot(Wm, Sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])

    # new covariance is the sum of the outer product of the residuals
    # times the weights
    P = zeros((n, n))
    for k in range(kmax):
        y = Sigmas[k] - x
        P += Wc[k] * np.outer(y, y)

    return (x, P + noise_cov)