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

from __future__ import (absolute_import, division)
from filterpy.common import dot3
import numpy as np
from numpy import dot, zeros


class IMMEstimator(object):
    """ Implements an Interacting Multiple-Model (IMM) estimator.
    """

    def __init__(self, x_shape, filters, p, trans):

        self.filters = filters
        self.w = p
        self.trans = trans
        try:
            self.N = x_shape[0]
        except:
            self.N = x_shape

        self.x = np.zeros(x_shape)
        self.P = np.zeros((self.N, self.N))

        self.cbar = dot(self.trans.T, self.w)
        self.n = len(filters)



    '''@property
    def x(self):
        """ The estimated state of the bank of filters."""
        return self._x

    @property
    def P(self):
        """ Estimated covariance of the bank of filters."""
        return self._P'''



    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        **Parameters**

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array,  or None
            Optionally provide H to override the measurement function for this
            one call, otherwise  self.H will be used.

        """

        L = zeros(len(self.filters))
        for i, f in enumerate(self.filters):
            f.update(z, R, H)
            L[i] = f.likelihood    # prior * likelihood

        # compute weights
        self.w =  self.cbar * L
        self.w /= sum(self.w) # normalize

        # compute IMM state and covariance
        self.x.fill(0.)
        self.P.fill(0.)

        for f, w in zip(self.filters, self.w):
            self.x += f.x*w

        for f, w in zip(self.filters, self.w):
            y = f.x - self.x
            self.P += w*(np.outer(y, y) + f.P)


        # initial condition IMM state, covariance
        xs, Ps = [], []
        self.cbar = dot(self.trans.T, self.w)

        omega = np.zeros((self.n, self.n))
        for i in range(self.n):
            omega[i, 0] = self.trans[i, 0]*self.w[i] / self.cbar[0]
            omega[i, 1] = self.trans[i, 1]*self.w[i] / self.cbar[1]


        # compute mixed states
        for i, (f, w) in enumerate(zip(self.filters, omega.T)):
            x = np.zeros(self.x.shape)
            P = np.zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                x += kf.x * wj

            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                P += wj*(np.outer(y, y) + kf.P)
            xs.append(x)
            Ps.append(P)

        for i in range(len(xs)):
            f = self.filters[i]
            # propagate using the mixed state estimate and covariance
            f.x = dot(f.F, xs[i])
            f.P = dot3(f.F, Ps[i], f.F.T) + f.Q
