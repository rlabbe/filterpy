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



def sigma_points (xm, P, kappa):
    """ Computes the sigma points and weights for an unscented Kalman filter.
    xm are the means, and P is the covariance. kappa is an arbitrary constant
    constant. Returns tuple of the sigma points and weights.

    Works with both scalar and array inputs:
    sigma_points (5, 9, 2) # mean 5, covariance 9
    sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
    """
#
    # check to see if len() is implemented is not bulletproof, but so long
    # as caller passes in either a scalar, numpy array, or numpy matrix
    # we will distinguish the two correct cases.
    if not hasattr(xm, '__len__'):
        xm = np.array([[xm]])

    if not hasattr(P, '__len__'):
        P = np.array([[P]])

    n = np.size(xm)
    Xi = np.zeros((n,2*n+1))
    W  = np.zeros(2*n+1)

    Xi[:,0] = xm.T

    W[0] = float(kappa) / (n + kappa)

    # U'*U = (n+kappa)*P
    U = linalg.cholesky((n+kappa)*P)

    for k in range (n):
        print (xm + U[:,k])
        Xi[:,k+1] = xm + U[:,k]
        W[k+1] = 1. / (2*(n+kappa))

    for k in range (n):
        Xi[:, n+k+1] = xm - U[:,k]
        W[n+k+1] = 1. / (2.*(n+kappa))

    return (Xi, W)


def unscented_transform (Xi, W, NoiseCov=None):
    """ computes the unscented transform of a set of signma points and weights.
    returns the mean and covariance in a tuple
    """
    n, kmax = Xi.shape
    xm = 0

    for k in range (kmax):
        xm += W[k] * Xi[:,k]

    cov = np.zeros((n,n))

    for k in range (kmax):
        cov += float(W[k])*(Xi[:,k]-xm).dot((Xi[:,k]-xm).T)

    if NoiseCov is not None:
        cov += NoiseCov
    return (xm, cov)

if __name__ == "__main__":
    Xi, W = sigma_points (5,9,2)
    xm, cov = unscented_transform(Xi, W)
    print(xm)
    print(cov)

    xm = np.array([[1],[2],[3]])
    P = np.eye(3) * 2.

    Xi, W = sigma_points (xm, P, 4)
    xm, cov = unscented_transform(Xi, W)
    print(xm)
    print (cov)
#    sigma_points ([5,2],9*np.eye(2), 2)

