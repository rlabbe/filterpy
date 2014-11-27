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

    .. [1] Julier, Simon J. "A New Extension of the Kalman Filter to Nonlinear
        Systems". Proc. SPIE 3068, Signal Processing, Sensor Fusion, and
        Target Recognition VI, 182 (July 28, 1997)

    .. [2] Labbe, Roger R. "Kalman and Bayesian Filters in Python"

        https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, dim_x, dim_z, dt, kappa=0.):
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
        self.xp = None
        self.Pp = None
        self._dim_x = dim_x
        self._dim_z = dim_z
        self._dt = dt
        self._num_sigmas = 2*dim_x + 1
        self.kappa = kappa

        # weights for the means and covariances. In this formation
        # both are the same.
        self.Wm = self.weights(dim_x, kappa)
        self.Wc = self.Wm

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update
        self.fX = zeros((2*self._dim_x+1, self._dim_x))
        self.hX = zeros((self._num_sigmas, self._dim_z))


    def update(self, z, hx, fx):
        """ Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter,
        and self.xp and self.Pp contain the prediction only, if that is
        of interest to you.


        **Parameters**

        z : numpy.array of shape (dim_z)
            measurement vector

        hz : function(x)
            Measurement function. Converts state vector x into a measurement
            vector of shape (dim_z).

        fx : function(x,dt)
            function that returns the state x transformed by the
            state transistion function. dt is the time step in seconds.
        """

        # rename for readability
        fX = self.fX
        hX = self.hX
        Wm = self.Wm
        Wc = self.Wc

        # Prediction step

        # calculate sigma points for given mean and covariance
        X = self.sigma_points(self.x, self.P, self.kappa)

        for i in range(self._num_sigmas):
            fX[i] = fx(X[i], self._dt)

        self.xp, self.Pp = unscented_transform(fX, Wm, Wc, self.Q)


        # update step

        # transform sigma points into measurement space
        for i in range(self._num_sigmas):
            hX[i] = hx(fX[i])

        # mean and covariance of prediction passed through UT
        zp, Pz = unscented_transform(hX, Wm, Wc, self.R)

        # compute cross variance of the state and the measurements
        Pxz = zeros((self._dim_x, self._dim_z))
        for i in range(self._num_sigmas):
            Pxz += Wc[i] * np.outer(fX[i] - self.xp, hX[i] - zp)

        K = dot(Pxz, inv(Pz)) # Kalman gain

        self.x = self.xp + dot(K, z-zp)
        self.P = self.Pp - dot3(K, Pz, K.T)


    @staticmethod
    def weights(n, kappa):
        """ Computes the weights for an unscented Kalman filter.  See
        __init__() for meaning of parameters.
        """

        k = 1. / (2*(n+kappa))
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

        Sigmas = zeros((2*n+1, n))

        # implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        U = cholesky((n+kappa)*P).T
        #U = sqrtm((n+kappa)*P).T

        for k in range(n):
            Sigmas[k+1]   = x + U[k]
            Sigmas[n+k+1] = x - U[k]

        # handle value for the mean separately as special case
        Sigmas[0] = x

        return Sigmas




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

    def __init__(self, dim_x, dim_z, dt, alpha, beta, kappa=0.):
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

        alpha : float
            Determins the spread of the sigma points around the mean.
            Usually a small positive value (1e-3) according to [1].

        beta : float
            Incorporates prior knowledge of the distribution of the mean. For
            Gaussian x beta=2 is optimal, according to [1].

        kappa : float, default=0.0
            Secondary scaling parameter usually set to 0 according to [2],
            or to 3-n according to [3].


        **References**

        .. [1] S. Julier, J. Uhlmann, and H. Durrant-Whyte. "A new method for
               the nonlinear transformation of means and covariances in filters
               and estimators," IEEE Transactions on Automatic Control, 45(3),
               pp. 477-482 (March 2000).

        .. [2] E. A. Wan and R. Van der Merwe, “The Unscented Kalman filter for
               Nonlinear Estimation,” in Proc. Symp. Adaptive Syst. Signal
               Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

               https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

        .. [3] Wan, Merle "The Unscented Kalman Filter," chapter in *Kalman
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

        # weights for the means and covariances. In this formation
        # both are the same.
        self.Wm, self.Wc = self.weights(dim_x, alpha, beta, kappa)


        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update
        self.fX = zeros((2*self._dim_x+1, self._dim_x))
        self.hX = zeros((self._num_sigmas, self._dim_z))


    def update(self, z, hx, fx):
        """ Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter,
        and self.xp and self.Pp contain the prediction only, if that is
        of interest to you.


        **Parameters**

        z : numpy.array of shape (dim_z)
            measurement vector

        hz : function(x)
            Measurement function. Converts state vector x into a measurement
            vector of shape (dim_z).

        fx : function(x,dt)
            function that returns the state x transformed by the
            state transistion function. dt is the time step in seconds.
        """

        # rename for readability
        fX = self.fX
        hX = self.hX
        Wm = self.Wm
        Wc = self.Wc

        #################
        # Prediction step

        # calculate sigma points for given mean and covariance
        X = self.sigma_points(self.x, self.P, self.alpha, self.kappa)

        for i in range(self._num_sigmas):
            fX[i] = fx(X[i], self._dt)

        self.xp, self.Pp = unscented_transform(fX, Wm, Wc, self.Q)

        #################
        # update step

        # transform sigma points into measurement space
        for i in range(self._num_sigmas):
            hX[i] = hx(fX[i])

        # mean and covariance of prediction passed through UT
        zp, Pz = unscented_transform(hX, Wm, Wc, self.R)

        # compute cross variance of the state and the measurements
        Pxz = zeros((self._dim_x, self._dim_z))
        for i in range(self._num_sigmas):
            Pxz += Wc[i] * np.outer(fX[i] - self.xp, hX[i] - zp)

        K = dot(Pxz, inv(Pz)) # Kalman gain

        # compute new estimates
        self.x = self.xp + dot(K, z-zp)
        self.P = self.Pp - dot3(K, Pz, K.T)


    @staticmethod
    def weights(n, alpha, beta, kappa):
        """ Computes the weights for the scaled unscented Kalman filter.
        """

        lambda_ = (alpha**2)*(n+kappa)-n

        c = 1. / (2*(n+lambda_))
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

        Sigmas = zeros((2*n+1, n)) # sigma points

        # efficient square root of matrix calculation. Implements
        #     U'*U = lambda_*P.
        # Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        lambda_ = alpha**2 * (n+kappa)
        U = cholesky(lambda_*P).T
        #U = sqrtm((lambda_)*P).T

        for k in range(n):
            Sigmas[k+1]   = x + U[k]
            Sigmas[n+k+1] = x - U[k]

        # handle value for the mean separately as special case
        Sigmas[0] = x

        return Sigmas


def unscented_transform(Xi, Wm, Wc, noise_cov):
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