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

from numpy.linalg import inv, cholesky
import numpy as np
from numpy import asarray, eye, zeros, dot
from filterpy.common import dot3
from filterpy.kalman import unscented_transform


class ScaledUnscentedKalmanFilter(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=C0103
    """ Implements the Scaled Unscented Kalman filter (UKF) as defined by
    Simon Julier in [1], using the formulation provided by Wan and Merle
    in [2]. This filter scales the sigma points to avoid strong nonlinearities.


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

    .. [1] Julier, Simon J. "The scaled unscented transformation,"
        American Control Converence, 2002, pp 4555-4559, vol 6.

        Online copy:
        https://www.cs.unc.edu/~welch/kalman/media/pdf/ACC02-IEEE1357.PDF


    .. [2] E. A. Wan and R. Van der Merwe, “The unscented Kalman filter for
        nonlinear estimation,” in Proc. Symp. Adaptive Syst. Signal
        Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

        Online Copy:
        https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    """

    def __init__(self, dim_x, dim_z, dt, alpha, hx, fx, beta, kappa=0.):
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

        alpha : float
            Determins the spread of the sigma points around the mean.
            Usually a small positive value (1e-3) according to [3].

        beta : float
            Incorporates prior knowledge of the distribution of the mean. For
            Gaussian x beta=2 is optimal, according to [3].

        kappa : float, default=0.0
            Secondary scaling parameter usually set to 0 according to [4],
            or to 3-n according to [5].


        **References**

        .. [3] S. Julier, J. Uhlmann, and H. Durrant-Whyte. "A new method for
               the nonlinear transformation of means and covariances in filters
               and estimators," IEEE Transactions on Automatic Control, 45(3),
               pp. 477-482 (March 2000).

        .. [4] E. A. Wan and R. Van der Merwe, “The Unscented Kalman filter for
               Nonlinear Estimation,” in Proc. Symp. Adaptive Syst. Signal
               Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

               https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

        .. [5] Wan, Merle "The Unscented Kalman Filter," chapter in *Kalman
               Filtering and Neural Networks*, John Wiley & Sons, Inc., 2001.
        """

        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.xp = None
        self.Pp = None
        self._dim_x = dim_x
        self._dim_z = dim_z
        self._dt = dt
        self._num_sigmas = 2*dim_x + 1
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.hx = hx
        self.fx = fx

        # weights for the means and covariances. In this formation
        # both are the same.
        self.Wm, self.Wc = self.weights(dim_x, alpha, beta, kappa)


        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update
        self.sigmas_f = zeros((2*self._dim_x+1, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))



    def predict(self):
        """ Performs the predict step of the UKF. On return, self.xp and
        self.Pp contain the predicted state (xp) and covariance (Pp). 'p'
        stands for prediction.

        Important: this MUST be called before update() is called for the first
        time.
        """

        # rename for readability
        Wm = self.Wm
        Wc = self.Wc

        # calculate sigma points for given mean and covariance
        sigmas = self.sigma_points(self.x, self.P, self.kappa)
        for i in range(self._num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], self._dt)

        self.xp, self.Pp = unscented_transform(self.sigmas_f, Wm, Wc, self.Q)


    def update(self, z, residual=np.subtract, UT=None):
        """ Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        **Parameters**

        z : numpy.array of shape (dim_z)
            measurement vector

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

        # rename for readability
        sigmas_f = self.sigmas_f
        sigmas_h = self.sigmas_h
        Wm = self.Wm
        Wc = self.Wc
        if UT is None:
            UT = unscented_transform

        # transform sigma points into measurement space
        sigmas_h2 = self.hx(sigmas_f)

        for i in range(self._num_sigmas):
            sigmas_h[i] = self.hx(sigmas_f[i])

        assert sigmas_h2.all() == sigmas_h.all()

        # mean and covariance of prediction passed through UT
        zp, Pz = UT(sigmas_h, Wm, Wc, self.R)

        # compute cross variance of the state and the measurements
        Pxz = zeros((self._dim_x, self._dim_z))
        for i in range(self._num_sigmas):
            Pxz += Wc[i] * np.outer(sigmas_f[i] - self.xp,
                                    residual(sigmas_h[i], zp))

        K = dot(Pxz, inv(Pz)) # Kalman gain
        y = residual(z, zp)   #residual

        self.x = self.xp + dot(K, y)
        self.P = self.Pp - dot3(K, Pz, K.T)


    @staticmethod
    def weights(n, alpha, beta, kappa):
        """ Computes the weights for the scaled unscented Kalman filter.
        """

        lambda_ = (alpha**2)*(n+kappa)-n

        c = .5 / (n+lambda_)
        Wc = np.full(2*n+1, c)
        Wm = np.full(2*n+1, c)
        Wc[0] = lambda_ / (n+lambda_) + (1 - alpha**2 + beta)
        Wm[0] = lambda_ / (n+lambda_)

        return Wm, Wc


    @staticmethod
    def sigma_points(x, P, alpha, kappa):
        """ Computes the sigma pointsfor an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        kappa is an arbitrary constant
        constant. Returns tuple of the sigma points and weights.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        **Parameters**

        X An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        alpha : float
            Determines the spread of the sigma points around the mean.

        kappa : float
            Scaling factor.

        **Returns**

        sigmas : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.

            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """

        if np.isscalar(x):
            x = asarray([x])
        n = np.size(x)  # dimension of problem

        if  np.isscalar(P):
            P = eye(n)*P

        sigmas = zeros((2*n+1, n)) # sigma points

        # efficient square root of matrix calculation. Implements
        #     U'*U = lambda_*P.
        # Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        lambda_ = alpha**2 * (n+kappa)
        U = cholesky(lambda_*P).T
        #U = sqrtm((lambda_)*P).T

        for k in range(n):
            sigmas[k+1]   = x + U[k]
            sigmas[n+k+1] = x - U[k]

        # handle value for the mean separately as special case
        sigmas[0] = x

        return sigmas
