# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 23:20:51 2014

@author: rlabbe

The test implement code from P. Zarchan Fundamentals of Kalman Filtering
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt


class LeastSquaresFilter0(object):
    
    def __init__(self):
        self.k = 0
        self.x = 0.
    
    def __call__(self, x):
         self.k += 1
             
         residual =  x - self.x
         self.x = self.x + residual/self.k
         
         return self.x
         
class LeastSquaresFilter1(object):
    
    def __init__(self, dt):
        self.k = 0
        self.x = 0.
        self.dx = 0
        self.dt = dt
    
    def __call__(self, x):
         self.k += 1
             
         K1 = 2*(2*self.k-1) / (self.k*(self.k+1))
         K2 = 6 / (self.k * (self.k+1) * self.dt)
         
         residual =  x - self.x - self.dx*self.dt
         self.x = self.x + self.dx*self.dt + K1 * residual
         self.dx = self.dx + K2*residual
         
         return self.x


class LeastSquaresFilter2(object):
    
    def __init__(self, dt):
        self.k = 0
        self.x = 0.
        self.dx = 0.
        self.ddx = 0.
        self.dt = dt
        self.dt2 = dt**2
    
    def __call__(self, x):
         self.k += 1
         
         k = self.k
         den = k*(k+1)*(k+2)
         K1 = 3*(3*k**2 - 3*k + 2) / den
         K2 = 18*(2*k-1) / (den*self.dt)
         K3 = 60./ (den*self.dt2)
         print(K1,K2,K3)
                         
         residual =  x - self.x - self.dx*self.dt - .5*self.ddx*self.dt2
         self.x += self.dx*self.dt + .5*self.ddx*self.dt2 + K1 * residual
         self.dx += self.ddx*self.dt + K2*residual
         self.ddx += K3*residual
         print(self.x, self.dx, self.ddx)
         return self.x
         
         
def test_first_order ():
    ''' data and example from Zarchan, page 105-6'''
    
    lsf = LeastSquaresFilter1(1)
    
    xs = [1.2, .2, 2.9, 2.1]
    ys = []
    for x in xs:
        ys.append (lsf(x))
    
    plt.plot(xs,c='b')
    plt.plot(ys, c='g')
    plt.plot([0,len(xs)-1], [ys[0], ys[-1]])

         
def test_second_order ():
    ''' data and example from Zarchan, page 114'''
    
    lsf = LeastSquaresFilter2(1)
    
    xs = [1.2, .2, 2.9, 2.1]
    ys = []
    for x in xs:
        ys.append (lsf(x))
    
    plt.plot(xs,c='b')
    plt.plot(ys, c='g')
    plt.plot([0,len(xs)-1], [ys[0], ys[-1]])
    
import numpy.random as random

def fig_3_8():
    """ figure 3.8 in Zarchan, p. 108"""
    lsf = LeastSquaresFilter1(0.1) 
    
    xs = [x+3 + random.randn() for x in np.arange (0,10, 0.1)]
    ys = []
    for x in xs:
        ys.append (lsf(x))
    
    plt.plot(xs)
    plt.plot(ys)    


def listing_3_4():
    """ listing 3.4 in Zarchan, p. 117"""
    
    lsf = LeastSquaresFilter2(0.1) 
    
    xs = [5*x*x -x + 2 + 30*random.randn() for x in np.arange (0,10, 0.1)]
    ys = []
    for x in xs:
        ys.append (lsf(x))
    
    plt.plot(xs)
    plt.plot(ys)
    
listing_3_4()

#test_second_order()
#fig_3_8()