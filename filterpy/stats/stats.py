# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments, bad-whitespace
# pylint: disable=too-many-lines, too-many-locals, len-as-condition

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


from __future__ import absolute_import, division, unicode_literals

import math
from math import cos, sin
import random
import warnings
import numpy as np
from numpy.linalg import inv
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spln
from scipy.stats import norm, multivariate_normal



# Older versions of scipy do not support the allow_singular keyword. I could
# check the version number explicily, but perhaps this is clearer
_support_singular = True
try:
    multivariate_normal.logpdf(1, 1, 1, allow_singular=True)
except TypeError:
    warnings.warn(
        'You are using a version of SciPy that does not support the '\
        'allow_singular parameter in scipy.stats.multivariate_normal.logpdf(). '\
        'Future versions of FilterPy will require a version of SciPy that '\
        'implements this keyword',
        DeprecationWarning)
    _support_singular = False


def _validate_vector(u, dtype=None):
    # this is taken from scipy.spatial.distance. Internal function, so
    # redefining here.

    u = np.asarray(u, dtype=dtype).squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


def mahalanobis(x, mean, cov):
    """
    Computes the Mahalanobis distance between the state vector x from the
    Gaussian `mean` with covariance `cov`. This can be thought as the number
    of standard deviations x is from the mean, i.e. a return value of 3 means
    x is 3 std from mean.

    Parameters
    ----------
    x : (N,) array_like, or float
        Input state vector

    mean : (N,) array_like, or float
        mean of multivariate Gaussian

    cov : (N, N) array_like  or float
        covariance of the multivariate Gaussian

    Returns
    -------
    mahalanobis : double
        The Mahalanobis distance between vectors `x` and `mean`

    Examples
    --------
    >>> mahalanobis(x=3., mean=3.5, cov=4.**2) # univariate case
    0.125

    >>> mahalanobis(x=3., mean=6, cov=1) # univariate, 3 std away
    3.0

    >>> mahalanobis([1., 2], [1.1, 3.5], [[1., .1],[.1, 13]])
    0.42533327058913922
    """

    x = _validate_vector(x)
    mean = _validate_vector(mean)

    if x.shape != mean.shape:
        raise ValueError("length of input vectors must be the same")

    y = x - mean
    S = np.atleast_2d(cov)

    dist = float(np.dot(np.dot(y.T, inv(S)), y))
    return math.sqrt(dist)


def log_likelihood(z, x, P, H, R):
    """
    Returns log-likelihood of the measurement z given the Gaussian
    posterior (x, P) using measurement function H and measurement
    covariance error R
    """
    S = np.dot(H, np.dot(P, H.T)) + R
    return logpdf(z, np.dot(H, x), S)


def likelihood(z, x, P, H, R):
    """
    Returns likelihood of the measurement z given the Gaussian
    posterior (x, P) using measurement function H and measurement
    covariance error R
    """
    return np.exp(log_likelihood(z, x, P, H, R))


def logpdf(x, mean=None, cov=1, allow_singular=True):
    """
    Computes the log of the probability density function of the normal
    N(mean, cov) for the data x. The normal may be univariate or multivariate.

    Wrapper for older versions of scipy.multivariate_normal.logpdf which
    don't support support the allow_singular keyword prior to verion 0.15.0.

    If it is not supported, and cov is singular or not PSD you may get
    an exception.

    `x` and `mean` may be column vectors, row vectors, or lists.
    """

    if mean is not None:
        flat_mean = np.asarray(mean).flatten()
    else:
        flat_mean = None

    flat_x = np.asarray(x).flatten()

    if _support_singular:
        return multivariate_normal.logpdf(flat_x, flat_mean, cov, allow_singular)
    return multivariate_normal.logpdf(flat_x, flat_mean, cov)


def gaussian(x, mean, var, normed=True):
    """
    returns normal distribution (pdf) for x given a Gaussian with the
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

    norm : bool, default True
        Normalize the output if the input is an array of values.

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

    g = ((2*math.pi*var)**-.5) * np.exp((-0.5*(np.asarray(x)-mean)**2.) / var)
    if normed and len(np.shape(g)) > 0:
        g = g / sum(g)

    return g



def mul(mean1, var1, mean2, var2):
    """
    Multiply Gaussian (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean, var).

    Strictly speaking the product of two Gaussian PDFs is a Gaussian
    function, not Gaussian PDF. It is, however, proportional to a Gaussian
    PDF, so it is safe to treat the output as a PDF for any filter using
    Bayes equation, which normalizes the result anyway.

    Parameters
    ----------
    mean1 : scalar
         mean of first Gaussian

    var1 : scalar
         variance of first Gaussian

    mean2 : scalar
         mean of second Gaussian

    var2 : scalar
         variance of second Gaussian

    Returns
    -------
    mean : scalar
        mean of product

    var : scalar
        variance of product

    Examples
    --------
    >>> mul(1, 2, 3, 4)
    (1.6666666666666667, 1.3333333333333333)

    References
    ----------
    Bromily. "Products and Convolutions of Gaussian Probability Functions",
    Tina Memo No. 2003-003.
    http://www.tina-vision.net/docs/memos/2003-003.pdf
    """

    mean = (var1*mean2 + var2*mean1) / (var1 + var2)
    var = 1 / (1/var1 + 1/var2)
    return (mean, var)


def mul_pdf(mean1, var1, mean2, var2):
    """
    Multiply Gaussian (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean, var, scale_factor).

    Strictly speaking the product of two Gaussian PDFs is a Gaussian
    function, not Gaussian PDF. It is, however, proportional to a Gaussian
    PDF. `scale_factor` provides this proportionality constant

    Parameters
    ----------
    mean1 : scalar
         mean of first Gaussian

    var1 : scalar
         variance of first Gaussian

    mean2 : scalar
         mean of second Gaussian

    var2 : scalar
         variance of second Gaussian

    Returns
    -------
    mean : scalar
        mean of product

    var : scalar
        variance of product

    scale_factor : scalar
        proportionality constant


    Examples
    --------
    >>> mul(1, 2, 3, 4)
    (1.6666666666666667, 1.3333333333333333)

    References
    ----------
    Bromily. "Products and Convolutions of Gaussian Probability Functions",
    Tina Memo No. 2003-003.
    http://www.tina-vision.net/docs/memos/2003-003.pdf
    """

    mean = (var1*mean2 + var2*mean1) / (var1 + var2)
    var = 1. / (1./var1 + 1./var2)

    S = math.exp(-(mean1 - mean2)**2 / (2*(var1 + var2))) / \
                 math.sqrt(2 * math.pi * (var1 + var2))

    return mean, var, S


def add(mean1, var1, mean2, var2):
    """
    Add the Gaussians (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean,var).

    var1 and var2 are variances - sigma squared in the usual parlance.
    """

    return (mean1+mean2, var1+var2)


def multivariate_gaussian(x, mu, cov):
    """
    This is designed to replace scipy.stats.multivariate_normal
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

    warnings.warn(
        ("This was implemented before SciPy version 0.14, which implemented "
         "scipy.stats.multivariate_normal. This function will be removed in "
         "a future release of FilterPy"), DeprecationWarning)

    # force all to numpy.array type, and flatten in case they are vectors
    x = np.array(x, copy=False, ndmin=1).flatten()
    mu = np.array(mu, copy=False, ndmin=1).flatten()

    nx = len(mu)
    cov = _to_cov(cov, nx)


    norm_coeff = nx*math.log(2*math.pi) + np.linalg.slogdet(cov)[1]

    err = x - mu
    if sp.issparse(cov):
        numerator = spln.spsolve(cov, err).T.dot(err)
    else:
        numerator = np.linalg.solve(cov, err).T.dot(err)

    return math.exp(-0.5*(norm_coeff + numerator))


def multivariate_multiply(m1, c1, m2, c2):
    """
    Multiplies the two multivariate Gaussians together and returns the
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
        Covariance of first Gaussian. Must be convertable to an 2D array via
        numpy.asarray().

     m2 : array-like
        Mean of second Gaussian. Must be convertable to an 1D array via
        numpy.asarray(), For example 6, [6], [6, 5], np.array([3, 4, 5, 6])
        are all valid.

    c2 : matrix-like
        Covariance of second Gaussian. Must be convertable to an 2D array via
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


def plot_discrete_cdf(xs, ys, ax=None, xlabel=None, ylabel=None,
                      label=None):
    """
    Plots a normal distribution CDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the cumulative probability.

    Parameters
    ----------

    xs : list-like of scalars
        x values corresponding to the values in `y`s. Can be `None`, in which
        case range(len(ys)) will be used.

    ys : list-like of scalars
        list of probabilities to be plotted which should sum to 1.

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend

    Returns
    -------
        axis of plot
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if xs is None:
        xs = range(len(ys))
    ys = np.cumsum(ys)
    ax.plot(xs, ys, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_gaussian_cdf(mean=0., variance=1.,
                      ax=None,
                      xlim=None, ylim=(0., 1.),
                      xlabel=None, ylabel=None,
                      label=None):
    """
    Plots a normal distribution CDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the cumulative probability.

    Parameters
    ----------

    mean : scalar, default 0.
        mean for the normal distribution.

    variance : scalar, default 0.
        variance for the normal distribution.

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend

    Returns
    -------
        axis of plot
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    sigma = math.sqrt(variance)
    n = norm(mean, sigma)
    if xlim is None:
        xlim = [n.ppf(0.001), n.ppf(0.999)]

    xs = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 1000.)
    cdf = n.cdf(xs)
    ax.plot(xs, cdf, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_gaussian_pdf(mean=0.,
                      variance=1.,
                      std=None,
                      ax=None,
                      mean_line=False,
                      xlim=None, ylim=None,
                      xlabel=None, ylabel=None,
                      label=None):
    """
    Plots a normal distribution PDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the probability density.

    Parameters
    ----------

    mean : scalar, default 0.
        mean for the normal distribution.

    variance : scalar, default 1., optional
        variance for the normal distribution.

    std: scalar, default=None, optional
        standard deviation of the normal distribution. Use instead of
        `variance` if desired

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    mean_line : boolean
        draws a line at x=mean

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend

    Returns
    -------
        axis of plot
    """

    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if variance is not None and std is not None:
        raise ValueError('Specify only one of variance and std')

    if variance is None and std is None:
        raise ValueError('Specify variance or std')

    if variance is not None:
        std = math.sqrt(variance)

    n = norm(mean, std)

    if xlim is None:
        xlim = [n.ppf(0.001), n.ppf(0.999)]

    xs = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 1000.)
    ax.plot(xs, n.pdf(xs), label=label)
    ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if mean_line:
        plt.axvline(mean)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def plot_gaussian(mean=0., variance=1.,
                  ax=None,
                  mean_line=False,
                  xlim=None,
                  ylim=None,
                  xlabel=None,
                  ylabel=None,
                  label=None):
    """
    DEPRECATED. Use plot_gaussian_pdf() instead. This is poorly named, as
    there are multiple ways to plot a Gaussian.
    """

    warnings.warn('This function is deprecated. It is poorly named. '\
                  'A Gaussian can be plotted as a PDF or CDF. This '\
                  'plots a PDF. Use plot_gaussian_pdf() instead,',
                  DeprecationWarning)
    return plot_gaussian_pdf(mean, variance, ax, mean_line, xlim, ylim, xlabel,
                             ylabel, label)


def covariance_ellipse(P, deviations=1):
    """
    Returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Parameters
    ----------

    P : nd.array shape (2,2)
       covariance matrix

    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.

    Returns (angle_radians, width_radius, height_radius)
    """

    U, s, _ = linalg.svd(P)
    orientation = math.atan2(U[1, 0], U[0, 0])
    width = deviations * math.sqrt(s[0])
    height = deviations * math.sqrt(s[1])

    if height > width:
        raise ValueError('width must be greater than height')

    return (orientation, width, height)


def _eigsorted(cov, asc=True):
    """
    Computes eigenvalues and eigenvectors of a covariance matrix and returns
    them sorted by eigenvalue.

    Parameters
    ----------
    cov : ndarray
        covariance matrix

    asc : bool, default=True
        determines whether we are sorted smallest to largest (asc=True),
        or largest to smallest (asc=False)

    Returns
    -------
    eigval : 1D ndarray
        eigenvalues of covariance ordered largest to smallest

    eigvec : 2D ndarray
        eigenvectors of covariance matrix ordered to match `eigval` ordering.
        I.e eigvec[:, 0] is the rotation vector for eigval[0]
    """

    eigval, eigvec = np.linalg.eigh(cov)
    order = eigval.argsort()
    if not asc:
        # sort largest to smallest
        order = order[::-1]

    return eigval[order], eigvec[:, order]


def plot_3d_covariance(mean, cov, std=1.,
                       ax=None, title=None,
                       color=None, alpha=1.,
                       label_xyz=True,
                       N=60,
                       shade=True,
                       limit_xyz=True,
                       **kwargs):
    """
    Plots a covariance matrix `cov` as a 3D ellipsoid centered around
    the `mean`.

    Parameters
    ----------

    mean : 3-vector
        mean in x, y, z. Can be any type convertable to a row vector.

    cov : ndarray 3x3
        covariance matrix

    std : double, default=1
        standard deviation of ellipsoid

    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Axis to draw on. If not provided, a new 3d axis will be generated
        for the current figure

    title : str, optional
        If provided, specifies the title for the plot

    color : any value convertible to a color
        if specified, color of the ellipsoid.

    alpha : float, default 1.
        Alpha value of the ellipsoid. <1 makes is semi-transparent.

    label_xyz: bool, default True

        Gives labels 'X', 'Y', and 'Z' to the axis.

    N : int, default=60
        Number of segments to compute ellipsoid in u,v space. Large numbers
        can take a very long time to plot. Default looks nice.

    shade : bool, default=True
        Use shading to draw the ellipse

    limit_xyz : bool, default=True
        Limit the axis range to fit the ellipse

    **kwargs : optional
        keyword arguments to supply to the call to plot_surface()
    """

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # force mean to be a 1d vector no matter its shape when passed in
    mean = np.atleast_2d(mean)
    if mean.shape[1] == 1:
        mean = mean.T

    if not(mean.shape[0] == 1 and mean.shape[1] == 3):
        raise ValueError('mean must be convertible to a 1x3 row vector')
    mean = mean[0]

    # force covariance to be 3x3 np.array
    cov = np.asarray(cov)
    if cov.shape[0] != 3 or cov.shape[1] != 3:
        raise ValueError("covariance must be 3x3")

    # The idea is simple - find the 3 axis of the covariance matrix
    # by finding the eigenvalues and vectors. The eigenvalues are the
    # radii (squared, since covariance has squared terms), and the
    # eigenvectors give the rotation. So we make an ellipse with the
    # given radii and then rotate it to the proper orientation.

    eigval, eigvec = _eigsorted(cov, asc=True)
    radii = std * np.sqrt(np.real(eigval))

    if eigval[0] < 0:
        raise ValueError("covariance matrix must be positive definite")


    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, N)
    v = np.linspace(0.0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v)) * radii[0]
    y = np.outer(np.sin(u), np.sin(v)) * radii[1]
    z = np.outer(np.ones_like(u), np.cos(v)) * radii[2]

    # rotate data with eigenvector and center on mu
    a = np.kron(eigvec[:, 0], x)
    b = np.kron(eigvec[:, 1], y)
    c = np.kron(eigvec[:, 2], z)

    data = a + b + c
    N = data.shape[0]
    x = data[:,   0:N]   + mean[0]
    y = data[:,   N:N*2] + mean[1]
    z = data[:, N*2:]    + mean[2]

    fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z,
                    rstride=3, cstride=3, linewidth=0.1, alpha=alpha,
                    shade=shade, color=color, **kwargs)

    # now make it pretty!

    if label_xyz:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    if limit_xyz:
        r = radii.max()
        ax.set_xlim(-r + mean[0], r + mean[0])
        ax.set_ylim(-r + mean[1], r + mean[1])
        ax.set_zlim(-r + mean[2], r + mean[2])

    if title is not None:
        plt.title(title)

    #pylint: disable=pointless-statement
    Axes3D #kill pylint warning about unused import

    return ax


def plot_covariance_ellipse(
        mean, cov=None, variance=1.0, std=None,
        ellipse=None, title=None, axis_equal=True, show_semiaxis=False,
        facecolor=None, edgecolor=None,
        fc='none', ec='#004080',
        alpha=1.0, xlim=None, ylim=None,
        ls='solid'):
    """
    Deprecated function to plot a covariance ellipse. Use plot_covariance
    instead.

    See Also
    --------

    plot_covariance
    """

    warnings.warn("deprecated, use plot_covariance instead", DeprecationWarning)
    plot_covariance(mean=mean, cov=cov, variance=variance, std=std,
                    ellipse=ellipse, title=title, axis_equal=axis_equal,
                    show_semiaxis=show_semiaxis, facecolor=facecolor,
                    edgecolor=edgecolor, fc=fc, ec=ec, alpha=alpha,
                    xlim=xlim, ylim=ylim, ls=ls)


def _std_tuple_of(var=None, std=None, interval=None):
    """
    Convienence function for plotting. Given one of var, standard
    deviation, or interval, return the std. Any of the three can be an
    iterable list.

    Examples
    --------
    >>>_std_tuple_of(var=[1, 3, 9])
    (1, 2, 3)

    """

    if std is not None:
        if np.isscalar(std):
            std = (std,)
        return std


    if interval is not None:
        if np.isscalar(interval):
            interval = (interval,)

        return norm.interval(interval)[1]

    if var is None:
        raise ValueError("no inputs were provided")

    if np.isscalar(var):
        var = (var,)
    return np.sqrt(var)


def plot_covariance(
        mean, cov=None, variance=1.0, std=None, interval=None,
        ellipse=None, title=None, axis_equal=True,
        show_semiaxis=False, show_center=True,
        facecolor=None, edgecolor=None,
        fc='none', ec='#004080',
        alpha=1.0, xlim=None, ylim=None,
        ls='solid'):
    """
    Plots the covariance ellipse for the 2D normal defined by (mean, cov)

    `variance` is the normal sigma^2 that we want to plot. If list-like,
    ellipses for all ellipses will be ploted. E.g. [1,2] will plot the
    sigma^2 = 1 and sigma^2 = 2 ellipses. Alternatively, use std for the
    standard deviation, in which case `variance` will be ignored.

    ellipse is a (angle,width,height) tuple containing the angle in radians,
    and width and height radii.

    You may provide either cov or ellipse, but not both.

    Parameters
    ----------

    mean : row vector like (2x1)
        The mean of the normal

    cov : ndarray-like
        2x2 covariance matrix

    variance : float, default 1, or iterable float, optional
        Variance of the plotted ellipse. May specify std or interval instead.
        If iterable, such as (1, 2**2, 3**2), then ellipses will be drawn
        for all in the list.


    std : float, or iterable float, optional
        Standard deviation of the plotted ellipse. If specified, variance
        is ignored, and interval must be `None`.

        If iterable, such as (1, 2, 3), then ellipses will be drawn
        for all in the list.

    interval : float range [0,1), or iterable float, optional
        Confidence interval for the plotted ellipse. For example, .68 (for
        68%) gives roughly 1 standand deviation. If specified, variance
        is ignored and `std` must be `None`

        If iterable, such as (.68, .95), then ellipses will be drawn
        for all in the list.


    ellipse: (float, float, float)
        Instead of a covariance, plots an ellipse described by (angle, width,
        height), where angle is in radians, and the width and height are the
        minor and major sub-axis radii. `cov` must be `None`.

    title: str, optional
        title for the plot

    axis_equal: bool, default=True
        Use the same scale for the x-axis and y-axis to ensure the aspect
        ratio is correct.

    show_semiaxis: bool, default=False
        Draw the semiaxis of the ellipse

    show_center: bool, default=True
        Mark the center of the ellipse with a cross

    facecolor, fc: color, default=None
        If specified, fills the ellipse with the specified color. `fc` is an
        allowed abbreviation

    edgecolor, ec: color, default=None
        If specified, overrides the default color sequence for the edge color
        of the ellipse. `ec` is an allowed abbreviation

    alpha: float range [0,1], default=1.
        alpha value for the ellipse

    xlim: float or (float,float), default=None
       specifies the limits for the x-axis

    ylim: float or (float,float), default=None
       specifies the limits for the y-axis

    ls: str, default='solid':
        line style for the edge of the ellipse
    """

    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt

    if cov is not None and ellipse is not None:
        raise ValueError('You cannot specify both cov and ellipse')

    if cov is None and ellipse is None:
        raise ValueError('Specify one of cov or ellipse')

    if facecolor is None:
        facecolor = fc

    if edgecolor is None:
        edgecolor = ec

    if cov is not None:
        ellipse = covariance_ellipse(cov)

    if axis_equal:
        plt.axis('equal')

    if title is not None:
        plt.title(title)

    ax = plt.gca()

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.
    height = ellipse[2] * 2.

    std = _std_tuple_of(variance, std, interval)
    for sd in std:
        e = Ellipse(xy=mean, width=sd*width, height=sd*height, angle=angle,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    lw=2, ls=ls)
        ax.add_patch(e)
    x, y = mean
    if show_center:
        plt.scatter(x, y, marker='+', color=edgecolor)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_semiaxis:
        a = ellipse[0]
        h, w = height/4, width/4
        plt.plot([x, x+ h*cos(a+np.pi/2)], [y, y + h*sin(a+np.pi/2)])
        plt.plot([x, x+ w*cos(a)], [y, y + w*sin(a)])


def norm_cdf(x_range, mu, var=1, std=None):
    """
    Computes the probability that a Gaussian distribution lies
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


def _to_cov(x, n):
    """
    If x is a scalar, returns a covariance matrix generated from it
    as the identity matrix multiplied by x. The dimension will be nxn.
    If x is already a 2D numpy array then it is returned unchanged.

    Raises ValueError if not positive definite
    """

    if np.isscalar(x):
        if x < 0:
            raise ValueError('covariance must be > 0')
        return np.eye(n) * x

    x = np.atleast_2d(x)
    try:
        # quickly find out if we are positive definite
        np.linalg.cholesky(x)
    except:
        raise ValueError('covariance must be positive definit')

    return x


def rand_student_t(df, mu=0, std=1):
    """
    return random number distributed by student's t distribution with
    `df` degrees of freedom with the specified mean and standard deviation.
    """

    x = random.gauss(0, std)
    y = 2.0*random.gammavariate(0.5 * df, 2.0)
    return x / (math.sqrt(y / df)) + mu


def NESS(xs, est_xs, ps):
    """
    Computes the normalized estimated error squared test on a sequence
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
