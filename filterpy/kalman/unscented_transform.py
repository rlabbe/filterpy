# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:57:08 2015

@author: Roger
"""

import numpy as np
from scipy.linalg import cholesky


def unscented_transform(sigmas, Wm, Wc, noise_cov=None):
    """ Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.

    Parameters
    ----------

    sigamas: ndarray [#sigmas per dimension, dimension]
        2D array of sigma points.

    Wm : ndarray [# sigmas per dimension]
        Weights for the mean. Must sum to 1.


    Wc : ndarray [# sigmas per dimension]
        Weights for the covariance. Must sum to 1.

    Noise_cov : ndarray, optional
        noise matrix added to the final computed covariance matrix.

    Returns
    -------

    x : ndarray [dimension]
        Mean of the sigma points after passing through the transform.

    P : ndarray
        covariance of the sigma points after passing throgh the transform.
    """

    kmax, n = sigmas.shape

    # new mean is just the sum of the sigmas * weight
    x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])

    # new covariance is the sum of the outer product of the residuals
    # times the weights
    '''P = zeros((n, n))
    for k in range(kmax):
        y = Sigmas[k] - x
        P += Wc[k] * np.outer(y, y)'''

    # this is the fast way to do the commented out code above
    y = sigmas - x[np.newaxis,:]
    P = y.T.dot(np.diag(Wc)).dot(y)

    if noise_cov is not None:
        P += noise_cov

    return (x, P)


class MerweScaledSigmaPoints(object):

    def __init__(self, n, alpha, beta, kappa, sqrt_method=cholesky):
        """
        Parameters
        ----------

        n : int
            Dimensionality of the state. 2n+1 weights will be generated.

        alpha : float
            Determins the spread of the sigma points around the mean.
            Usually a small positive value (1e-3) according to [3].

        beta : float
            Incorporates prior knowledge of the distribution of the mean. For
            Gaussian x beta=2 is optimal, according to [3].

        kappa : float, default=0.0
            Secondary scaling parameter usually set to 0 according to [4],
            or to 3-n according to [5].

        sqrt_method : function(ndarray), default=scipy.linalg.cholesky
            Defines how we compute the square root of a matrix, which has
            no unique answer. Cholesky is the default choice due to its
            speed. Typically your alternative choice will be
            scipy.linalg.sqrtm. Different choices affect how the sigma points
            are arranged relative to the eigenvectors of the covariance matrix.
            Usually this will not matter to you; if so the default cholesky()
            yields maximal performance. As of van der Merwe's dissertation of
            2004 [6] this was not a well reseached area so I have no advice
            to give you.

            If your method returns a triangular matrix it must be upper
            triangular. Do not use numpy.linalg.cholesky - for historical
            reasons it returns a lower triangular matrix. The SciPy version
            does the right thing.
        """

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.sqrt = sqrt_method


    def sigma_points(self, x, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        Parameters
        ----------

        X An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        Returns
        -------

        sigmas : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.

            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """

        assert self.n == np.size(x), "expected size {}, but size is {}".format(
            self.n, np.size(x))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(n)*P

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        sigmas = np.zeros((2*n+1, n))
        U = self.sqrt((lambda_ + n)*P)

        for k in range(n):
            sigmas[k+1]   = x + U[k]
            sigmas[n+k+1] = x - U[k]

        # handle value for the mean separately as special case
        sigmas[0] = x

        return sigmas



    def weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.

        Returns
        -------
        Wm : ndarray[2n+1]
            weights for mean

        Wc : ndarray[2n+1]
            weights for the covariances
        """

        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        Wc = np.full(2*n + 1, c)
        Wm = np.full(2*n + 1, c)
        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        Wm[0] = lambda_ / (n + lambda_)

        return Wm, Wc


class JulierSigmaPoints(object):

    def __init__(self,n,  kappa, sqrt_method=cholesky):
        """
        Parameters
        ----------

        n : int
            Dimensionality of the state. 2n+1 weights will be generated.

        kappa : float, default=0.
            Scaling factor that can reduce high order errors. kappa=0 gives
            the standard unscented filter. According to [Julier], if you set
            kappa to 3-dim_x for a Gaussian x you will minimize the fourth
            order errors in x and P.

        sqrt_method : function(ndarray), default=scipy.linalg.cholesky
            Defines how we compute the square root of a matrix, which has
            no unique answer. Cholesky is the default choice due to its
            speed. Typically your alternative choice will be
            scipy.linalg.sqrtm. Different choices affect how the sigma points
            are arranged relative to the eigenvectors of the covariance matrix.
            Usually this will not matter to you; if so the default cholesky()
            yields maximal performance. As of van der Merwe's dissertation of
            2004 [6] this was not a well reseached area so I have no advice
            to give you.

            If your method returns a triangular matrix it must be upper
            triangular. Do not use numpy.linalg.cholesky - for historical
            reasons it returns a lower triangular matrix. The SciPy version
            does the right thing.
       """

        self.n = n
        self.kappa = kappa
        self.sqrt = sqrt_method


    def sigma_points(self, x, P):
        r""" Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        kappa is an arbitrary constant. Returns sigma points.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        Parameters
        ----------

        X An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        kappa : float
            Scaling factor.

        **Returns**

        sigmas : np.array, of size (n, 2n+1)
            2D array of sigma points :math:`\chi`. Each column contains all of
            the sigmas for one dimension in the problem space. They
            are ordered as:

            .. math::
                :nowrap:

                \begin{eqnarray}
                  \chi[0]    = &x \\
                  \chi[1..n] = &x + [\sqrt{(n+\kappa)P}]_k \\
                  \chi[n+1..2n] = &x - [\sqrt{(n+\kappa)P}]_k
                \end{eqnarray}
        """

        assert self.n == np.size(x)
        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        n = np.size(x)  # dimension of problem

        if np.isscalar(P):
            P = np.eye(n)*P

        sigmas = np.zeros((2*n+1, n))

        # implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
        # Take transpose so we can access with U[i]
        U = self.sqrt((n + self.kappa) * P)

        sigmas[0] = x
        sigmas[1:n+1]     = x + U
        sigmas[n+1:2*n+2] = x - U
        return sigmas


    def weights(self):
        """ Computes the weights for the unscented Kalman filter.


        Returns
        -------
        Wm : ndarray[2n+1]
            weights for mean

        Wc : ndarray[2n+1]
            weights for the covariances
        """
        n = self.n
        k = self.kappa

        k = .5 / (n + k)
        W = np.full(2*n+1, k)
        W[0] = k / (n+k)
        return W, W
