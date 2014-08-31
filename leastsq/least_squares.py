# -*- coding: utf-8 -*-

"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.

The test implement code from P. Zarchan Fundamentals of Kalman Filtering
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


class LeastSquaresFilter(object):
    def __init__(self, dt, order, noise_variance=0.):
        """ Least Squares filter of order 0 to 2.
        
        Parameters
        ----------
        dt : float
           time step per update
           
        order : int
            order of filter 0..2
            
        noise_variance : float
            variance in x
        """
        
        assert order >= 0
        assert order <= 2

        self.reset()

        self.dt = dt
        self.dt2 = dt**2

        self.sigma = noise_variance
        self._order = order
        
        


    def reset(self):
        """ reset filter back to state at time of construction"""

        self.n = 0 #nth step in the recursion
        self.x = 0.
        self.error = 0.
        self.derror = 0.
        self.dderror = 0.
        self.dx = 0.
        self.ddx = 0.
        self.K1 = 0
        self.K2 = 0
        self.K3 = 0


    def __call__(self, z):
        self.n += 1
        n = self.n
        dt = self.dt
        dt2 = self.dt2

        if self._order == 0:
            self.K1 = 1./n
            residual =  z - self.x
            self.x = self.x + residual * self.K1
            self.error = self.sigma/sqrt(n)

        elif self._order == 1:
            self.K1 = 2*(2*n-1) / (n*(n+1))
            self.K2 = 6 / (n*(n+1)*dt)

            residual =  z - self.x - self.dx*dt
            self.x = self.x + self.dx*dt + self.K1*residual
            self.dx = self.dx + self.K2*residual

            if n > 1:
                self.error = self.sigma*sqrt(2.*(2*n-1)/(n*(n+1)))
                self.derror = self.sigma*sqrt(12./(n*(n*n-1)*dt*dt))

        else:
            den = n*(n+1)*(n+2)
            self.K1 = 3*(3*n**2 - 3*n + 2) / den
            self.K2 = 18*(2*n-1) / (den*dt)
            self.K3 = 60./ (den*dt2)

            residual =  z - self.x - self.dx*dt - .5*self.ddx*dt2
            self.x   += self.dx*dt  + .5*self.ddx*dt2 +self. K1 * residual
            self.dx  += self.ddx*dt + self.K2*residual
            self.ddx += self.K3*residual

            if n >= 3:
                self.error = self.sigma*sqrt(3*(3*n*n-3*n+2)/(n*(n+1)*(n+2)))
                self.derror = self.sigma*sqrt(12*(16*n*n-30*n+11) /
                                              (n*(n*n-1)*(n*n-4)*dt2))
                self.dderror = self.sigma*sqrt(720/(n*(n*n-1)*(n*n-4)*dt2*dt2))

        return self.x


    def standard_deviation(self):
        if self.n == 0:
            return 0.

        if self._order == 0:
            return 1./sqrt(self)

        elif self._order == 1:
            pass


    def __repr__(self):
        return 'LeastSquareFilter x={}, dx={}, ddx={}'.format(
               self.x, self.dx, self.ddx)


class LeastSquaresFilterGH(object):
    """ Implements the Least Squares filter using the g-h filter.
    This is lighter weight than the LeastSquaresFilter cla"""