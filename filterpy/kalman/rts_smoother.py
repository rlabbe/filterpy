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

from numpy.linalg import inv
from numpy import dot, zeros
from filterpy.common import dot3


def rts_smoother(Xs, Ps, F, Q):
    """ Runs the Rauch-Tung-Striebal Kalman smoother on a set of
    means and covariances computed by a Kalman filter. The usual input
    would come from the output of `KalmanFilter.batch_filter()`.

    **Parameters**

    Xs : numpy.array
       array of the means (state variable x) of the output of a Kalman
       filter.

    Ps : numpy.array
        array of the covariances of the output of a kalman filter.

    F : numpy.array
        State transition function of the Kalman filter

    Q : numpy.array
        Process noise of the Kalman filter


    **Returns**

    'x' : numpy.ndarray
       smoothed means

    'P' : numpy.ndarray
       smoothed state covariances

    'K' : numpy.ndarray
        smoother gain at each step


    **Example**::

        zs = [t + random.randn()*4 for t in range (40)]

        (mu, cov, _, _) = kalman.batch_filter(zs)
        (x, P, K) = rks_smoother(mu, cov, fk.F, fk.Q)

    """

    assert len(Xs) == len(Ps)
    shape = Xs.shape
    n = shape[0]
    dim_x = shape[1]

    # smoother gain
    K = zeros((n,dim_x,dim_x))

    x, P = Xs.copy(), Ps.copy()

    for k in range(n-2,-1,-1):
        P_pred = dot3(F, P[k], F.T) + Q

        K[k]  = dot3(P[k], F.T, inv(P_pred))
        x[k] += dot (K[k], x[k+1] - dot(F, x[k]))
        P[k] += dot3 (K[k], P[k+1] - P_pred, K[k].T)

    return (x, P, K)
