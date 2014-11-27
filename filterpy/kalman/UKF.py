# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

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
#from scipy.linalg import sqrtm


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

    There are multiple formulations for computing the sigma points and
    weights. This class supports multiple versions via external classes.
    Several are provided, and you can write your own. Follow the same
    method signature as implemented by JulierPoints.


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

        [1] Julier, Simon J. "A New Extension of the Kalman Filter to Nonlinear
            Systems". Proc. SPIE 3068, Signal Processing, Sensor Fusion, and
            Target Recognition VI, 182 (July 28, 1997)

        [2] Labbe, Roger R. "Kalman and Bayesian Filters in Python"

            https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, dim_x, dim_z, dt, points_alg):
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

        points_alg : class
            Object that computes weights and sigma points. There are several
            methods in the literature; choose one of the classes below
            (such as WanMerlePoints) depending on the algorithm you want.
            WanMerlePoints is the standard implementation in most of the
            recent literature.

        """

        assert len(points_alg.Wc) == 2*dim_x + 1

        self._points_alg = points_alg

        self._sigma_points = points_alg.sigma_points

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

        self.Wc = points_alg.Wc
        self.Wm = points_alg.Wm

        # sigma points transformed through f(x)
        self._fXi = zeros((2*self._dim_x+1, self._dim_x))


    def predict(self, fx):
        """ Perform predict step of the UKF. Note that this
        MUST be called before update, order of predict() and
        update() is not optional as it is in the kalman filter.

        This function has the side effect of modifying self.xp and self.Pp,
        the predicted mean and covariance.

        ***Parameters**

        fx : function(x,dt)
            function that returns the state x transformed by the
            state transistion function. dt is the time step in seconds.
        """

        # calculate sigma points for given mean and covariance
        Xi = self._sigma_points(self.x, self.P)

        for i in range(self._num_sigmas):
            self._fXi[i] = fx(Xi[i], self._dt)

        self.xp, self.Pp = self.unscented_transform(self._fXi,
                                                    self.Wm, self.Wc, self.Q)


    def update(self, z, hx):
        """ Perform the update step of the UKF.

        this function has the side effect of modifying self.x and self.P,
        the estimated mean and covariance.

        **Parameters**

        z : numpy.array of shape (dim_z)
            measurement vector

        hz : function(x)
            Measurement function. Converts state vector x into a measurement
            vector of shape (dim_z).
        """

        hXi = zeros((self._num_sigmas, self._dim_z))
        for i in range(self._num_sigmas):
            hXi[i] = hx(self._fXi[i])

        # mean and covariance of measurment passed through UT
        zp, Pz = self.unscented_transform(hXi, self.Wm, self.Wc, self.R)

        Pxz = zeros((self._dim_x, self._dim_z))

        for i in range(self._num_sigmas):
            Pxz += self.Wc[i] * np.outer(self._fXi[i] - self.xp, hXi[i] - zp)

        K = dot(Pxz, inv(Pz)) # Kalman gain

        self.x = self.xp + dot(K, z-zp)
        self.P = self.Pp - dot3(K, Pz, K.T)


    @staticmethod
    def unscented_transform(Xi, Wc, Wm, noise_cov):
        """ Computes unscented transform of a set of sigma points and weights.
        returns the mean and covariance in a tuple.
        """

        kmax, n = Xi.shape

        # new mean is just the sum of the sigmas * weight
        X = dot(Wm, Xi)    # \Sigma^n_1 (W[k]*Xi[k])

        # new covariance is the sum of the outer product of the residuals
        # times the weights
        P = zeros((n, n))
        for k in range(kmax):
            y = Xi[k] - X
            P += Wc[k] * np.outer(y, y)

        return (X, P + noise_cov)


class JulierPoints(object):
    # pylint: disable=C0103

    """ Computes sigma points and weights for for an unscented Kalman
    filter based on the original Julier97[1] paper.

    **References**

    [1] Julier, Simon J. "A New Extension of the Kalman Filter to Nonlinear
        Systems". Proc. SPIE 3068, Signal Processing, Sensor Fusion, and
        Target Recognition VI, 182 (July 28, 1997)
    """


    def __init__(self, n, kappa):
        """ Computes sigma points and weights for for an unscented Kalman
        filter.

        **Parameters**

        n : float
            Dimensions in the state vector

        kappa : float
            Scaling parameter.
        """
        self.W = self.weights(n, kappa)
        self.kappa = kappa


    @property
    def Wm(self):
        """ Weights for the means."""
        return self.W


    @property
    def Wc(self):
        """ Weights for the covariances."""
        return self.W


    @staticmethod
    def weights(n, kappa):
        """ Computes the weights for an unscented Kalman filter.  See
        __init__() for meaning of parameters.
        """

        k = 1. / (2*(n+kappa))
        W = np.full(2*n+1, k)
        W[0] = kappa / (n+kappa)
        return W


    def sigma_points(self, x, P, kappa=None):
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

        if kappa is None:
            kappa = self.kappa

        # Xi - sigma points
        Xi = zeros((2*n+1, n))

        # implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        U = cholesky((n+kappa)*P).T
        #U = sqrtm((n+kappa)*P).T

        for k in range(n):
            Xi[k+1]   = x + U[k]
            Xi[n+k+1] = x - U[k]

        # handle value for the mean separately as special case
        Xi[0] = x

        return Xi


class WanMerlePoints(object):
    # pylint: disable=C0103

    """ Computes sigma points and weights for for an unscented Kalman
    filter based on the Wan and Merle [1] paper.

    **References**

    [1] Wan, Merle "The Unscented Kalman Filter," chapter in *Kalman
        Filtering and Neural Networks*, John Wiley & Sons, Inc., 2001.
    """


    def __init__(self, n, alpha, beta, kappa):
        """ Computes sigma points and weights for for an unscented Kalman
        filter based on the original Wan and Merle formulation [1].

        **Parameters**

        n : float
            Dimensions in the state vector

        alpha : float
            Determins the spread of the sigma points around the mean.
            Usually a small positive value (1e-3) according to [1].

        beta : float
            Incorporates prior knowledge of the distribution of the mean. For
            Gaussian distributions beta=2 is optimal, according to [1].

        k : float
            Secondary scaling parameter usually set to 0 according to [1],
            or to 3-n according to [2].


        **References**

        [1] E. A. Wan and R. Van der Merwe, “The unscented Kalman filter for
            nonlinear estimation,” in Proc. Symp. Adaptive Syst. Signal
            Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

            https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

        [2] Wan, Merle "The Unscented Kalman Filter," chapter in *Kalman
            Filtering and Neural Networks*, John Wiley & Sons, Inc., 2001.

        """

        self.wc, self.wm = self.weights(n, alpha, beta, kappa)
        self.alpha = alpha
        self.kappa = kappa

    @property
    def Wm(self):
        """Weights for the means."""
        return self.wm


    @property
    def Wc(self):
        """weights for the covariances."""
        return self.wc


    @staticmethod
    def weights(n, alpha, beta, kappa):
        """ Computes the weights for an unscented Kalman filter. This is the
        formulation used by Wan and Merwe[1]. See __init__() for meaning of
        parameters.
        """

        lambda_ = (alpha**2)*(n+kappa)-n

        c = 1. / (2*(n+lambda_))
        Wc = np.full(2*n+1, c)
        Wm = np.full(2*n+1, c)
        Wc[0] = lambda_ / (n+lambda_) + (1 - alpha**2 + beta)
        Wm[0] = lambda_ / (n+lambda_)

        return Wc, Wm

    def sigma_points(self, x, P, alpha=None, kappa=None):
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

        if alpha is None:
            alpha = self.alpha

        if kappa is None:
            kappa = self.kappa

        Xi = zeros((2*n+1, n)) # sigma points

        lambda_ = alpha**2 * (n+kappa)

        # efficient square root of matrix calculation. Implements
        #     U'*U = lambda_*P.
        # Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        U = cholesky(lambda_*P).T
        #U = sqrtm((lambda_)*P).T

        for k in range(n):
            Xi[k+1]   = x + U[k]
            Xi[n+k+1] = x - U[k]

        # handle value for the mean separately as special case
        Xi[0] = x


        return Xi
