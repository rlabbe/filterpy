# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

Rauch-Tung-Striebal Kalman smoother from the filterpy library.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy.linalg import inv
from numpy import dot, zeros
from filterpy.common import dot3



def rks_smoother(Xs, Ps, F, Q, Xs_p=None, Ps_p=None):
    """ Runs the Rauch-Tung-Striebal Kalman smoother on a set of
    means and covariances computed by a Kalman filter. The usual input
    would come from the output of `KalmanFilter.batch_filter()`.

    Parameters
    ----------

    Xs : numpy.array
       array of the means (state variable x) of the output of a Kalman
       filter.

    Ps : numpy.array
        array of the covariances of the output of a kalman filter.

    F : numpy.array
        State transition function of the Kalman filter

    Q : numpy.array
        Process noise of the Kalman filter


    Returns
    -------
    'X' : numpy.ndarray
       smoothed means

    'P' : numpy.ndarray
       smoothed state covariances

    'C' : numpy.ndarray
        smoother gain at each step


    Example
    -------

    zs = [t + random.randn()*4 for t in range (40)]

    (mu, cov, _, _) = kalman.batch_filter(zs)
    (X, P, C) = rks_smoother(mu, cov, fk.F, fk.Q)


    """
    assert len(Xs) == len(Ps)
    n, dim_x, _ = Xs.shape

    # smoother gain
    C = zeros((n,dim_x,dim_x))

    X = Xs.copy()
    P = Ps.copy()

    if Xs_p is not None:
        assert Ps_p is not None

        for k in range(n-2,-1,-1):

            C[k] = dot3(P[k], F.T, inv(Ps_p[k]))
            X[k] = X[k] + dot (C[k], Xs_k[k] - dot(F, X[k]))
            P[k] = P[k] + dot3 (C[k], P[k+1] - Ps_p[k], C[k].T)


    else:

        for k in range(n-2,-1,-1):
            P_pred = dot3(F, P[k], F.T) + Q

            C[k] = dot3(P[k], F.T, inv(P_pred))
            X[k] = X[k] + dot (C[k], X[k+1] - dot(F, X[k]))
            P[k] = P[k] + dot3 (C[k], P[k+1] - P_pred, C[k].T)

    return (X,P,C)
