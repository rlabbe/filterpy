# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

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
        """ Least Squares filter of order 0 to 2."""

        assert order >= 0
        assert order <= 2

        self.k = 0
        self.x = 0.
        self.error = 0.
        self.derror = 0.
        self.dderror = 0.
        self.dx = 0.
        self.ddx = 0.
        self.dt = dt
        self.dt2 = dt**2
        
        self.sigma = noise_variance

        self._order = order


    def __call__(self, x):
        self.k += 1
        k = self.k
        if self._order == 0:
            residual =  x - self.x
            self.x = self.x + residual/k
            self.error = self.sigma/math.sqrt(k)

        elif self._order == 1:
            K1 = 2*(2*k-1) / (k*(k+1))
            K2 = 6 / (k * (k+1) * self.dt)

            residual =  x - self.x - self.dx*self.dt
            self.x = self.x + self.dx*self.dt + K1 * residual
            self.dx = self.dx + K2*residual
            
            self.error = self.sigma*sqrt(2.*(2*k-1)/(k*(k+1)))
            self.derror = self.sigma*sqrt(12./(k*(k*k-1)*self.dt*self.dt))

        else:
            den = k*(k+1)*(k+2)
            K1 = 3*(3*k**2 - 3*k + 2) / den
            K2 = 18*(2*k-1) / (den*self.dt)
            K3 = 60./ (den*self.dt2)
            dt = self.dt
            dt2 = self.dt2

            residual =  x - self.x - self.dx*dt - .5*self.ddx*self.dt2
            self.x   += self.dx*dt + .5*self.ddx*dt2 + K1 * residual
            self.dx  += self.ddx*dt + K2*residual
            self.ddx += K3*residual
            
            if k < 3:
                self.error = 0.
                self.derror = 0.
                self.dderror = 0.
            else:           
                self.error = self.sigma*sqrt(3*(3*k*k-3*k+2)/(k*(k+1)*(k+2)))
                self.derror = self.sigma*sqrt(12*(16*k*k-30*k+11) /
                                              (k*(k*k-1)*(k*k-4)*dt2))                                          
                self.dderror = self.sigma*sqrt(720/(k*(k*k-1)*(k*k-4)*dt2*dt2))

        return self.x
     
    def standard_deviation(self):
        if self.k == 0:
            return 0.
            
        if self._order == 0:
            return 1./math.sqrt(self)

        elif self._order == 1:
            pass            
        
