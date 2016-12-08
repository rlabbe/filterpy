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


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
from math import cos, sin
import numpy as np
import random
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import scipy.stats
from scipy.stats import norm
import warnings


def gaussian(x, mean, var):
    """returns normal distribution (pdf) for x given a Gaussian with the
    specified mean and variance. All must be scalars.

    gaussian (1,2,3) is equivalent to scipy.stats.norm(2,math.sqrt(3)).pdf(1)
    It is quite a bit faster albeit much less flexible than the latter.

    Parameters
    ----------

    x : scalar or array-like
        The value for which we compute the probability

    mean : scalar
        Mean of the Gaussian

    var : scalar
        Variance of the Gaussian

    Returns
    -------

    probability : float
        probability of x for the Gaussian (mean, var). E.g. 0.101 denotes
        10.1%.

    Examples
    --------

    >>> gaussian(8, 1, 2)
    1.3498566943461957e-06

    >>> gaussian([8, 7, 9], 1, 2)
    array([1.34985669e-06, 3.48132630e-05, 3.17455867e-08])
    """

    return (np.exp((-0.5*(np.asarray(x)-mean)**2)/var) /
            math.sqrt(2*math.pi*var))


def mul (mean1, var1, mean2, var2):
    """ multiply Gaussians (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean,var).

    var1 and var2 are variances - sigma squared in the usual parlance.
    """

    mean = (var1*mean2 + var2*mean1) / (var1 + var2)
    var = 1 / (1/var1 + 1/var2)
    return (mean, var)


def add (mean1, var1, mean2, var2):
    """ add the Gaussians (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean,var).

    var1 and var2 are variances - sigma squared in the usual parlance.
    """

    return (mean1+mean2, var1+var2)


def multivariate_gaussian(x, mu, cov):
    """ This is designed to replace scipy.stats.multivariate_normal
    which is not available before version 0.14. You may either pass in a
    multivariate set of data:

    .. code-block:: Python

       multivariate_gaussian (array([1,1]), array([3,4]), eye(2)*1.4)
       multivariate_gaussian (array([1,1,1]), array([3,4,5]), 1.4)

    or unidimensional data:

    .. code-block:: Python

       multivariate_gaussian(1, 3, 1.4)

    In the multivariate case if cov is a scalar it is interpreted as eye(n)*cov

    The function gaussian() implements the 1D (univariate)case, and is much
    faster than this function.

    equivalent calls:

    .. code-block:: Python

      multivariate_gaussian(1, 2, 3)
       scipy.stats.multivariate_normal(2,3).pdf(1)


    Parameters
    ----------

    x : float, or np.array-like
       Value to compute the probability for. May be a scalar if univariate,
       or any type that can be converted to an np.array (list, tuple, etc).
       np.array is best for speed.

    mu :  float, or np.array-like
       mean for the Gaussian . May be a scalar if univariate,  or any type
       that can be converted to an np.array (list, tuple, etc).np.array is
       best for speed.

    cov :  float, or np.array-like
       Covariance for the Gaussian . May be a scalar if univariate,  or any
       type that can be converted to an np.array (list, tuple, etc).np.array is
       best for speed.

    Returns
    -------

    probability : float
        probability for x for the Gaussian (mu,cov)
    """

    # force all to numpy.array type, and flatten in case they are vectors
    x   = np.array(x, copy=False, ndmin=1).flatten()
    mu  = np.array(mu,copy=False, ndmin=1).flatten()

    nx = len(mu)
    cov = _to_cov(cov, nx)

    norm_coeff = nx*math.log(2*math.pi) + np.linalg.slogdet(cov)[1]

    err = x - mu
    if (sp.issparse(cov)):
        numerator = spln.spsolve(cov, err).T.dot(err)
    else:
        numerator = np.linalg.solve(cov, err).T.dot(err)

    return math.exp(-0.5*(norm_coeff + numerator))


def multivariate_multiply(m1, c1, m2, c2):
    """ Multiplies the two multivariate Gaussians together and returns the
    results as the tuple (mean, covariance).

    Examples
    --------

    .. code-block:: Python

        m, c = multivariate_multiply([7.0, 2], [[1.0, 2.0], [2.0, 1.0]],
                                     [3.2, 0], [[8.0, 1.1], [1.1,8.0]])

    Parameters
    ----------

    m1 : array-like
        Mean of first Gaussian. Must be convertable to an 1D array via
        numpy.asarray(), For example 6, [6], [6, 5], np.array([3, 4, 5, 6])
        are all valid.

    c1 : matrix-like
        Mean of first Gaussian. Must be convertable to an 2D array via
        numpy.asarray().

     m2 : array-like
        Mean of second Gaussian. Must be convertable to an 1D array via
        numpy.asarray(), For example 6, [6], [6, 5], np.array([3, 4, 5, 6])
        are all valid.

    c2 : matrix-like
        Mean of second Gaussian. Must be convertable to an 2D array via
        numpy.asarray().

    Returns
    -------

    m : ndarray
        mean of the result

    c : ndarray
        covariance of the result
    """

    C1 = np.asarray(c1)
    C2 = np.asarray(c2)
    M1 = np.asarray(m1)
    M2 = np.asarray(m2)

    sum_inv = np.linalg.inv(C1+C2)
    C3 = np.dot(C1, sum_inv).dot(C2)

    M3 = (np.dot(C2, sum_inv).dot(M1) +
          np.dot(C1, sum_inv).dot(M2))

    return M3, C3


def covariance_ellipse(P, deviations=1):
    """ returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Parameters
    ----------

    P : nd.array shape (2,2)
       covariance matrix

    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.

    Returns (angle_radians, width_radius, height_radius)
    """

    U,s,v = linalg.svd(P)
    orientation = math.atan2(U[1,0],U[0,0])
    width  = deviations*math.sqrt(s[0])
    height = deviations*math.sqrt(s[1])

    assert width >= height
    return (orientation, width, height)

def norm_cdf (x_range, mu, var=1, std=None):
    """ computes the probability that a Gaussian distribution lies
    within a range of values.

    Parameters
    ----------

    x_range : (float, float)
        tuple of range to compute probability for

    mu : float
        mean of the Gaussian

    var : float, optional
        variance of the Gaussian. Ignored if `std` is provided

    std : float, optional
       standard deviation of the Gaussian. This overrides the `var` parameter

    Returns
    -------

    probability : float
        probability that Gaussian is within x_range. E.g. .1 means 10%.
    """

    if std is None:
        std = math.sqrt(var)
    return abs(norm.cdf(x_range[0], loc=mu, scale=std) -
               norm.cdf(x_range[1], loc=mu, scale=std))


def _is_inside_ellipse(x, y, ex, ey, orientation, width, height):

    co = np.cos(orientation)
    so = np.sin(orientation)

    xx = x*co + y*so
    yy = y*co - x*so

    return (xx / width)**2 + (yy / height)**2 <= 1.


    return ((x-ex)*co - (y-ey)*so)**2/width**2 + \
           ((x-ex)*so + (y-ey)*co)**2/height**2 <= 1

def _to_cov(x,n):
    """ If x is a scalar, returns a covariance matrix generated from it
    as the identity matrix multiplied by x. The dimension will be nxn.
    If x is already a numpy array then it is returned unchanged.
    """
    try:
        x.shape
        if type(x) != np.ndarray:
            x = np.asarray(x)[0]
        return x
    except:
        return np.eye(n) * x

def rand_student_t(df, mu=0, std=1):
    """return random number distributed by student's t distribution with
    `df` degrees of freedom with the specified mean and standard deviation.
    """
    x = random.gauss(0, std)
    y = 2.0*random.gammavariate(0.5*df, 2.0)
    return x / (math.sqrt(y/df)) + mu


def NESS(xs, est_xs, ps):
    """ Computes the normalized estimated error squared test on a sequence
    of estimates. The estimates are optimal if the mean error is zero and
    the covariance matches the Kalman filter's covariance. If this holds,
    then the mean of the NESS should be equal to or less than the dimension
    of x.

    Examples
    --------

    .. code-block: Python

        xs = ground_truth()
        est_xs, ps, _, _ = kf.batch_filter(zs)
        NESS(xs, est_xs, ps)

    Parameters
    ----------

    xs : list-like
        sequence of true values for the state x

    est_xs : list-like
        sequence of estimates from an estimator (such as Kalman filter)

    ps : list-like
        sequence of covariance matrices from the estimator

    Returns
    -------

    ness : list of floats
       list of NESS computed for each estimate

    """

    est_err = xs - est_xs
    ness = []
    for x, p in zip(est_err, ps):
        ness.append(np.dot(x.T, linalg.inv(p)).dot(x))
    return ness

