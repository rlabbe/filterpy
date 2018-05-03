# -*- coding: utf-8 -*-
# pylint: disable=C0103, R0913, R0902, C0326, R0914
# disable snake_case warning, too many arguments, too many attributes,
# one space before assignment, too many local variables

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

from __future__ import absolute_import, division
from math import sqrt
import numpy as np
from filterpy.kalman import pretty_str


class LeastSquaresFilter(object):
    """Implements a Least Squares recursive filter. Formulation is per
    Zarchan [1]_.

    Filter may be of order 0 to 2. Order 0 assumes the value being tracked is
    a constant, order 1 assumes that it moves in a line, and order 2 assumes
    that it is tracking a second order polynomial.


    Parameters
    ----------

    dt : float
       time step per update

    order : int
        order of filter 0..2

    noise_sigma : float
        sigma (std dev) in x. This allows us to calculate the error of
        the filter, it does not influence the filter output.


    Attributes
    ----------

    n : int
        step in the recursion. 0 prior to first call, 1 after the first call,
        etc.

    K1,K2,K3 : float
        Gains for the filter. K1 for all orders, K2 for orders 0 and 1, and
        K3 for order 2

    x, dx, ddx: type(z)
        estimate(s) of the output. 'd' denotes derivative, so 'dx' is the first
        derivative of x, 'ddx' is the second derivative.

    Examples
    --------

    .. code-block:: Python

        from filterpy.leastsq import LeastSquaresFilter

        lsq = LeastSquaresFilter(dt=0.1, order=1, noise_sigma=2.3)

        while True:
            z = sensor_reading()  # get a measurement
            x = lsq.update(z)     # get the filtered estimate.
            print('error: {}, velocity error: {}'.format(lsq.error, lsq.derror))

    References
    ----------

    .. [1] Zarchan and Musoff. "Fundamentals of Kalman Filtering: A Practical
          Approach." Third Edition. AIAA, 2009.
    """


    def __init__(self, dt, order, noise_sigma=0.):
        if order < 0 or order > 2:
            raise ValueError('order must be between 0 and 2')

        self.dt = dt
        self.dt2 = dt**2

        self.sigma = noise_sigma
        self._order = order

        self.reset()


    def reset(self):
        """ reset filter back to state at time of construction"""

        self.n = 0 #nth step in the recursion
        self.x = np.zeros(self._order + 1)
        self.K = np.zeros(self._order + 1)


    def update(self, z):
        """ Update filter with new measurement `z` """

        self.n += 1
        n = self.n
        dt = self.dt
        dt2 = self.dt2

        if self._order == 0:
            self.K[0] = 1./n
            residual = z - self.x
            self.x += residual * self.K[0]

        elif self._order == 1:
            self.K[0] = 2*(2*n-1) / (n*(n+1))
            self.K[1] = 6 / (n*(n+1)*dt)

            self.x[0] += self.x[1] * dt

            residual = z - self.x[0]

            self.x[0] += self.K[0] * residual
            self.x[1] += self.K[1] * residual

        else:
            den = n*(n+1)*(n+2)
            self.K[0] = 3*(3*n**2 - 3*n + 2) / den
            self.K[1] = 18*(2*n-1) / (den*dt)
            self.K[2] = 60./ (den*dt2)


            self.x[0] += self.x[1]*dt + .5*self.x[2]*dt2
            self.x[1] += self.x[2]*dt

            residual = z - self.x[0]

            self.x[0] += self.K[0] * residual
            self.x[1] += self.K[1] * residual
            self.x[2] += self.K[2] * residual

        return self.x


    def errors(self):
        """
        Computes and returns the error and  standard deviation  of the
        filter at this time step.

        Returns
        -------

        error : np.array size 1xorder+1
        std : np.array size 1xorder+1
        """

        n = self.n
        dt = self.dt
        order = self._order
        sigma = self.sigma

        error = np.zeros(order + 1)
        std = np.zeros(order + 1)


        if n == 0:
            return (error, std)

        if order == 0:
            error[0] = sigma/sqrt(n)
            std[0] = sigma/sqrt(n)

        elif order == 1:
            if n > 1:
                error[0] = sigma * sqrt(2*(2*n-1) / (n*(n+1)))
                error[1] = sigma * sqrt(12. / (n*(n*n-1)*dt*dt))
            std[0] = sigma * sqrt((2*(2*n-1)) / (n*(n+1)))
            std[1] = (sigma/dt) * sqrt(12. / (n*(n*n-1)))

        elif order == 2:
            dt2 = self.dt2

            if n >= 3:
                error[0] = sigma * sqrt(3*(3*n*n-3*n+2) / (n*(n+1)*(n+2)))
                error[1] = sigma * sqrt(12*(16*n*n-30*n+11) /
                                        (n*(n*n-1)*(n*n-4)*dt2))
                error[2] = sigma * sqrt(720/(n*(n*n-1)*(n*n-4)*dt2*dt2))

            std[0] = sigma * sqrt((3*(3*n*n - 3*n + 2)) / (n*(n+1)*(n+2)))
            std[1] = (sigma/dt) * sqrt((12*(16*n*n - 30*n + 11)) /
                                       (n*(n*n - 1)*(n*n -4)))
            std[2] = (sigma/dt2) * sqrt(720 / (n*(n*n-1)*(n*n-4)))

        return error, std


    def __repr__(self):
        return '\n'.join([
            'LeastSquaresFilter object',
            pretty_str('dt', self.dt),
            pretty_str('dt2', self.dt2),
            pretty_str('sigma', self.sigma),
            pretty_str('_order', self._order),
            pretty_str('x', self.x),
            pretty_str('K', self.K)
            ])
