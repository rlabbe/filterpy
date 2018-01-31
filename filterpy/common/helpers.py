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

import numpy as np
from numpy.linalg import inv


def _validate_vector(u, dtype=None):
    # this is taken from scipy.spatial.distance. Internal function, so
    # redefining here.

    u = np.asarray(u, dtype=dtype).squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


def runge_kutta4(y, x, dx, f):
    """computes 4th order Runge-Kutta for dy/dx.

    Parameters
    ----------

    y : scalar
        Initial/current value for y
    x : scalar
        Initial/current value for x
    dx : scalar
        difference in x (e.g. the time step)
    f : ufunc(y,x)
        Callable function (y, x) that you supply to compute dy/dx for
        the specified values.

    """

    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5*k1, x + 0.5*dx)
    k3 = dx * f(y + 0.5*k2, x + 0.5*dx)
    k4 = dx * f(y + k3, x + dx)

    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.


def mahalanobis(x, mean, cov):
    """ Computes the Mahalanobis distance between the state vector x from the
    Gaussian  `mean` with covariance `cov`.


    Parameters
    ----------
    x : (N,) array_like
        Input state vector
    mean : (N,) array_like
        mean of multivariate Gaussian
    cov : ndarray
        covariance of the multivariate Gaussian


    Returns
    -------
    mahalanobis : double
        The Mahalanobis distance between vectors `x` and `mean`


    Examples
    --------
    >>> mahalanobis(x=3., mean=3.5, cov=4.**2) # univariate case
    0.125
    >>> mahalanobis([1., 2], [1.1, 3.5], [[1., .1],[.1, 13]])
    0.42533327058913922
    """

    x = _validate_vector(x)
    mean = _validate_vector(mean)

    if x.shape != mean.shape:
        raise ValueError("length of input vectors must be the same")

    y = x - mean
    S = np.atleast_2d(cov)

    # residual y is now 1D, so no need to transpose, and y @S^ @ y is a scalar
    # so no array indexing required to get result
    dist = np.dot(y, inv(S)).dot(y)
    return np.sqrt(dist)

