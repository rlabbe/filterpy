# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

Rauch-Tung-Striebal Kalman smoother from the filterpy library.

filterpy library.
http:\\github.com\rlabbe\filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import numpy.linalg as linalg

class RKSSmoother(object):
    """ Rauch-Tung-Striebal Kalman smoother.

    Computes a smoothed sequence from a set of measurements.
    """

    def __init__(self):
        pass


    def smooth(self, Ms, Ps, F, Q):
        """ Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by a Kalman filter. The usual input
        would come from the output of `KalmanFilter.batch_filter()`.

        Parameters
        ----------

        Ms : numpy.array
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
        'M' : numpy.array
           smoothed means

        'P' : numpy.array
           smoothed state covariances

        'D' : numpy.array

        """
        M = np.copy(Ms)
        P = np.copy(Ps)
        assert len(M) == len(P)

        n     = np.size(M,0)  # number of measurements
        dim_x = np.size(M,1)  # number of state variables

        D = np.zeros((n,dim_x,dim_x))


        for k in range(n-2,-1,-1):
            P_pred = F.dot(P[k]).dot(F.T) + Q
            #D[k,:,:] = linalg.solve(P[k].dot(F.T).T, P_pred.T)
            D[k,:,:] = P[k].dot(linalg.solve((F.T).T, P_pred.T))
            M[k] = M[k] + D[k].dot(M[k+1] - F.dot(M[k]))
            P[k,:,:] = P[k,:,:] + D[k].dot(P[k+1,:,:] - P_pred).dot(D[k].T)


        return (M,P,D)

