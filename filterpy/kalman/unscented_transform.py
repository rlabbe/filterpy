# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments

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

import numpy as np


def unscented_transform(sigmas, Wm, Wc, noise_cov=None,
                        mean_fn=None, residual_fn=None):
    r"""
    Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.

    This works in conjunction with the UnscentedKalmanFilter class.


    Parameters
    ----------

    sigmas: ndarray, of size (n, 2n+1)
        2D array of sigma points.

    Wm : ndarray [# sigmas per dimension]
        Weights for the mean.


    Wc : ndarray [# sigmas per dimension]
        Weights for the covariance.

    noise_cov : ndarray, optional
        noise matrix added to the final computed covariance matrix.

    mean_fn : callable (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this if your state variable contains nonlinear
        values such as angles which cannot be summed.

        .. code-block:: Python

            def state_mean(sigmas, Wm):
                x = np.zeros(3)
                sum_sin, sum_cos = 0., 0.

                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += sin(s[2])*Wm[i]
                    sum_cos += cos(s[2])*Wm[i]
                x[2] = atan2(sum_sin, sum_cos)
                return x

    residual_fn : callable (x, y), optional

        Function that computes the residual (difference) between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.

        .. code-block:: Python

            def residual(a, b):
                y = a[0] - b[0]
                y = y % (2 * np.pi)
                if y > np.pi:
                    y -= 2*np.pi
                return y

    Returns
    -------

    x : ndarray [dimension]
        Mean of the sigma points after passing through the transform.

    P : ndarray
        covariance of the sigma points after passing throgh the transform.

    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    kmax, n = sigmas.shape

    try:
        if mean_fn is None:
            # new mean is just the sum of the sigmas * weight
            x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])
        else:
            x = mean_fn(sigmas, Wm)
    except:
        print(sigmas)
        raise


    # new covariance is the sum of the outer product of the residuals
    # times the weights

    # this is the fast way to do this - see 'else' for the slow way
    if residual_fn is np.subtract or residual_fn is None:
        y = sigmas - x[np.newaxis, :]
        P = np.dot(y.T, np.dot(np.diag(Wc), y))
    else:
        P = np.zeros((n, n))
        for k in range(kmax):
            y = residual_fn(sigmas[k], x)
            P += Wc[k] * np.outer(y, y)

    if noise_cov is not None:
        P += noise_cov

    return (x, P)
