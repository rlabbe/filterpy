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

import numpy.linalg as linalg
from numpy import dot, zeros
from filterpy.common import dot3


            
def rks_smoother(Xs, Ps, F, Q):
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
    

    """
    X = Xs.copy()
    P = Ps.copy()
    assert len(X) == len(P)

    n, dim_x, _ = X.shape

    # smoother gain
    C = zeros((n,dim_x,dim_x))

    for k in range(n-2,-1,-1):
        P_pred = dot3(F, P[k], F.T) + Q

        C[k] = dot3(P[k], F.T, linalg.inv(P_pred))
        X[k] = X[k] + dot (C[k], X[k+1] - dot(F, X[k]))
        P[k] = P[k] + dot3 (C[k], P[k+1] - P_pred, C[k].T)

    return (X,P,C)
