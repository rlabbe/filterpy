# -*- coding: utf-8 -*-
# pylint: disable=C0103, R0913, R0902, C0326, R0914, R0903
# disable snake_case warning, too many arguments, too many attributes,
# one space before assignment, too many local variables, too few public
# methods

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
import numpy as np
from numpy import dot
from filterpy.common import pretty_str


class FadingMemoryFilter(object):

    """ Creates a fading memory filter of order 0, 1, or 2.

    The KalmanFilter class also implements a more general fading memory
    filter and should be preferred in most cases. This is probably faster
    for low order systems.

    This algorithm is based on the fading filter algorithm developed in
    Zarcan's "Fundamentals of Kalman Filtering" [1].

    Parameters
    ----------

    x0 : 1D np.array or scalar
        Initial value for the filter state. Each value can be a scalar
        or a np.array.

        You can use a scalar for x0. If order > 0, then 0.0 is assumed
        for the higher order terms.

        x[0] is the value being tracked
        x[1] is the first derivative (for order 1 and 2 filters)
        x[2] is the second derivative (for order 2 filters)

    dt : scalar
        timestep

    order : int
        order of the filter. Defines the order of the system
        0 - assumes system of form x = a_0 + a_1*t
        1 - assumes system of form x = a_0 +a_1*t + a_2*t^2
        2 - assumes system of form x = a_0 +a_1*t + a_2*t^2 + a_3*t^3

    beta : float
        filter gain parameter.

    Attributes
    ----------

    x : np.array
        State of the filter.
        x[0] is the value being tracked
        x[1] is the derivative of x[0] (order 1 and 2 only)
        x[2] is the 2nd derivative of x[0] (order 2 only)

        This is always an np.array, even for order 0 where you can
        initialize x0 with a scalar.

    P : np.array
        The diagonal of the covariance matrix. Assumes that variance
        is one; multiply by sigma^2 to get the actual variances.

        This is a constant and will not vary as the filter runs.

    e : np.array
        The truncation error of the filter. Each term must be multiplied
        by the a_1, a_2, or a_3 of the polynomial for the system.

        For example, if the filter is order 2, then multiply all terms
        of self.e by a_3 to get the actual error. Multipy by a_2 for order
        1, and a_1 for order 0.


    References
    ----------

    Paul Zarchan and Howard Musoff. "Fundamentals of Kalman Filtering:
    A Practical Approach" American Institute of Aeronautics and Astronautics,
    Inc. Fourth Edition. p. 521-536. (2015)
    """

    def __init__(self, x0, dt, order, beta):

        if order < 0 or order > 2:
            raise ValueError('order must be between 0 and 2')

        if np.isscalar(x0):
            self.x = np.zeros(order+1)
            self.x[0] = x0
        else:
            self.x = np.copy(x0)

        self.dt = dt
        self.order = order
        self.beta = beta

        if order == 0:
            self.P = np.array([(1-beta)/(1+beta)], dtype=float)
            self.e = np.array([dt * beta / (1-beta)], dtype=float)

        elif order == 1:
            p11 = (1-beta) * (1+4*beta+5*beta**2) / (1+beta)**3
            p22 = 2*(1-beta)**3 / (1+beta)**3
            self.P = np.array([p11, p22], dtype=float)

            e = 2*dt*2 * (beta / (1-beta))**2
            de = dt*((1+3*beta)/(1-beta))
            self.e = np.array([e, de], dtype=float)

        else:
            p11 = (1-beta)*((1+6*beta + 16*beta**2 + 24*beta**3 + 19*beta**4) /
                            (1+beta)**5)

            p22 = (1-beta)**3 * ((13+50*beta + 49*beta**2) /
                                 (2*(1+beta)**5 * dt**2))

            p33 = 6*(1-beta)**5 / ((1+beta)**5 * dt**4)

            self.P = np.array([p11, p22, p33], dtype=float)

            e = 6*dt**3*(beta/(1-beta))**3
            de = dt**2 * (2 + 5*beta + 11*beta**2) / (1-beta)**2
            dde = 6*dt*(1+2*beta) / (1-beta)

            self.e = np.array([e, de, dde], dtype=float)

    def __repr__(self):
        return '\n'.join([
            'FadingMemoryFilter object',
            pretty_str('dt', self.dt),
            pretty_str('order', self.order),
            pretty_str('beta', self.beta),
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('e', self.e),
            ])


    def update(self, z):
        """ update the filter with measurement z. z must be the same type
        (or treatable as the same type) as self.x[0].
        """

        if self.order == 0:
            G = 1 - self.beta
            self.x = self.x + dot(G, (z - self.x))

        elif self.order == 1:
            G = 1 - self.beta**2
            H = (1-self.beta)**2
            x = self.x[0]
            dx = self.x[1]
            dxdt = dot(dx, self.dt)

            residual = z - (x+dxdt)
            self.x[0] = x + dxdt + G*residual
            self.x[1] = dx + (H / self.dt)*residual

        else: # order == 2
            G = 1-self.beta**3
            H = 1.5*(1+self.beta)*(1-self.beta)**2
            K = 0.5*(1-self.beta)**3

            x = self.x[0]
            dx = self.x[1]
            ddx = self.x[2]
            dxdt = dot(dx, self.dt)
            T2 = self.dt**2.

            residual = z - (x + dxdt + 0.5*ddx*T2)

            self.x[0] = x + dxdt + 0.5*ddx*T2 + G*residual
            self.x[1] = dx + ddx*self.dt + (H/self.dt)*residual
            self.x[2] = ddx + (2*K/(self.dt**2))*residual
