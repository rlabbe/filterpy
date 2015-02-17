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

import math
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import scipy.stats
from scipy.stats import norm
from matplotlib.patches import Ellipse
import random


def gaussian(x, mean, var):
    """returns normal distribution (pdf) for x given a Gaussian with the
    specified mean and variance. All must be scalars.

    gaussian (1,2,3) is equivalent to scipy.stats.norm(2,math.sqrt(3)).pdf(1)
    It is quite a bit faster albeit much less flexible than the latter.

    **Parameters**

    x : scalar
        The value for which we compute the probability

    mean : scalar
        Mean of the Gaussian

    var : scalar
        Variance of the Gaussian

    **Returns**

    probability : float
        probability of x for the Gaussian (mean, var). E.g. 0.101 denotes
        10.1%.
    """

    return math.exp((-0.5*(x-mean)**2)/var) / math.sqrt(2*math.pi*var)


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

       multivariate_gaussian (array([1,1]), array([3,4]), eye(2)*1.4)
       multivariate_gaussian (array([1,1,1]), array([3,4,5]), 1.4)

    or unidimensional data:
       multivariate_gaussian(1, 3, 1.4)

    In the multivariate case if cov is a scalar it is interpreted as eye(n)*cov

    The function gaussian() implements the 1D (univariate)case, and is much
    faster than this function.

    equivalent calls:
       multivariate_gaussian(1, 2, 3)
       scipy.stats.multivariate_normal(2,3).pdf(1)


    **Parameters**

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

    **Returns**

    probability : float
        probability for x for the Gaussian (mu,cov)
    """

    scipy.stats.multivariate_normal
    # force all to numpy.array type
    x   = np.array(x, copy=False, ndmin=1)
    mu  = np.array(mu,copy=False, ndmin=1)

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

    **example**

    m, c = multivariate_multiply([7.0, 2], [[1.0, 2.0], [2.0, 1.0]],
                                 [3.2, 0], [[8.0, 1.1], [1.1,8.0]])

    **Parameters**

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

    **Returns**

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



def plot_gaussian(mean, variance,
                  mean_line=False,
                  xlim=None,
                  xlabel=None,
                  ylabel=None):
    """ plots the normal distribution with the given mean and variance. x-axis
    contains the mean, the y-axis shows the probability.

    **parameters**

    mean_line : boolean
        draws a line at x=mean

    xlim: (float,float), optional
        specify the limits for the x axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis
    """

    sigma = math.sqrt(variance)
    n = scipy.stats.norm(mean, sigma)

    if xlim is None:
        min_x = n.ppf(0.001)
        max_x = n.ppf(0.999)
    else:
        min_x = xlim[0]
        max_x = xlim[1]
    xs = np.arange(min_x, max_x, (max_x - min_x) / 1000)
    plt.plot(xs,n.pdf(xs))
    plt.xlim((min_x, max_x))

    if mean_line:
        plt.axvline(mean)
    if xlabel:
       plt.xlabel(xlabel)
    if ylabel:
       plt.ylabel(ylabel)


def covariance_ellipse(P, deviations=1):
    """ returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    **Parameters**

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


def plot_covariance_ellipse(mean, cov=None,
            variance = 1.0, std=None,
             ellipse=None, title=None, axis_equal=True,
             facecolor='none', edgecolor='#004080',
             alpha=1.0, xlim=None, ylim=None):
    """ plots the covariance ellipse where

    mean is a (x,y) tuple for the mean of the covariance (center of ellipse)

    cov is a 2x2 covariance matrix.

    variance is the normal sigma^2 that we want to plot. If list-like,
    ellipses for all ellipses will be ploted. E.g. [1,2] will plot the
    sigma^2 = 1 and sigma^2 = 2 ellipses.

    ellipse is a (angle,width,height) tuple containing the angle in radians,
    and width and height radii.

    You may provide either cov or ellipse, but not both.

    plt.show() is not called, allowing you to plot multiple things on the
    same figure.
    """

    assert cov is None or ellipse is None
    assert not (cov is None and ellipse is None)

    if cov is not None:
        ellipse = covariance_ellipse(cov)

    if axis_equal:
        plt.axis('equal')

    if title is not None:
        plt.title (title)


    compute_std = False
    if std is None:
        std = variance
        compute_std = True


    if np.isscalar(std):
            std = [std]

    if compute_std:
        std = np.sqrt(np.asarray(std))

    ax = plt.gca()

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.
    height = ellipse[2] * 2.

    for sd in std:
        e = Ellipse(xy=mean, width=sd*width, height=sd*height, angle=angle,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    lw=2)
        ax.add_patch(e)
    plt.scatter(mean[0], mean[1], marker='+') # mark the center
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)


def norm_cdf (x_range, mu, var=1, std=None):
    """ computes the probability that a Gaussian distribution lies
    within a range of values.

    **Paramateters**

    x_range : (float, float)
        tuple of range to compute probability for

    mu : float
        mean of the Gaussian

    var : float, optional
        variance of the Gaussian. Ignored if std is provided

    std : float, optional
       standard deviation of the Gaussian. This overrides the var parameter


    **Returns**

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


def _do_plot_test():

    from numpy.random import multivariate_normal
    p = np.array([[32, 15],[15., 40.]])

    x,y = multivariate_normal(mean=(0,0), cov=p, size=5000).T
    sd = 2
    a,w,h = covariance_ellipse(p,sd)
    print (np.degrees(a), w, h)

    count = 0
    color=[]
    for i in range(len(x)):
        if _is_inside_ellipse(x[i], y[i], 0, 0, a, w, h):
            color.append('b')
            count += 1
        else:
            color.append('r')
    plt.scatter(x,y,alpha=0.2, c=color)


    plt.axis('equal')

    plot_covariance_ellipse(mean=(0., 0.),
                            cov = p,
                            std=sd,
                            facecolor='none')

    print (count / len(x))


def plot_std_vs_var():
    plt.figure()
    x = (0,0)
    P = np.array([[3,1],[1,3]])
    plot_covariance_ellipse(x, P, std=[1,2,3], facecolor='g', alpha=.2)
    plot_covariance_ellipse(x, P, variance=[1,2,3], facecolor='r', alpha=.5)


def rand_student_t(df, mu=0, std=1):
    """return random number distributed by student's t distribution with
    `df` degrees of freedom with the specified mean and standard deviation.
    """
    x = random.gauss(0, std)
    y = 2.0*random.gammavariate(0.5*df, 2.0)
    return x / (math.sqrt(y/df)) + mu


if __name__ == '__main__':
    plot_std_vs_var()
    plt.figure()

    _do_plot_test()

    #test_gaussian()

    # test conversion of scalar to covariance matrix
    x  = multivariate_gaussian(np.array([1,1]), np.array([3,4]), np.eye(2)*1.4)
    x2 = multivariate_gaussian(np.array([1,1]), np.array([3,4]), 1.4)
    assert x == x2

    # test univarate case
    rv = norm(loc = 1., scale = np.sqrt(2.3))
    x2 = multivariate_gaussian(1.2, 1., 2.3)
    x3 = gaussian(1.2, 1., 2.3)

    assert rv.pdf(1.2) == x2
    assert abs(x2- x3) < 0.00000001

    cov = np.array([[1.0, 1.0],
                    [1.0, 1.1]])

    plt.figure()
    P = np.array([[2,0],[0,2]])
    plot_covariance_ellipse((2,7), cov=cov, variance=[1,2], facecolor='g', title='my title', alpha=.2)
    plt.show()

    print("all tests passed")
