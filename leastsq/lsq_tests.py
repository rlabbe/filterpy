# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http:\\github.com\rlabbe\filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import numpy.random as random
import matplotlib.pyplot as plt

from filterpy.leastsq import LeastSquaresFilter

from filterpy.gh import GHFilter



def test_lsq():
    """ implements alternative version of first order Least Squares filter
    using g-h filter formulation and uses it to check the output of the
    LeastSquaresFilter class."""
    
    gh = GHFilter(x=0, dx=0, dt=1, g=.5, h=0.02)
    lsq = LeastSquaresFilter(dt=1, order=1)
    zs = [x+random.randn() for x in range(0,100)]

    xs = []
    lsq_xs= []
    for i,z in enumerate(zs):
        g = 2*(2*i + 1) / ((i+2)*(i+1))
        h = 6 / ((i+2)*(i+1))
        
        
        x,dx = gh.update(z,g,h)
        lsq_xs.append(lsq(z))
        xs.append(x)
        
        
    plt.plot(xs)
    plt.plot(lsq_xs)
    
    for x,y in zip(xs, lsq_xs):
        r = x-y
        assert r < 1.e-8
    

def test_first_order ():
    ''' data and example from Zarchan, page 105-6'''

    lsf = LeastSquaresFilter(dt=1, order=1)

    xs = [1.2, .2, 2.9, 2.1]
    ys = []
    for x in xs:
        ys.append (lsf(x))

    plt.plot(xs,c='b')
    plt.plot(ys, c='g')
    plt.plot([0,len(xs)-1], [ys[0], ys[-1]])


def test_second_order ():
    ''' data and example from Zarchan, page 114'''

    lsf = LeastSquaresFilter(1,order=2)

    xs = [1.2, .2, 2.9, 2.1]
    ys = []
    for x in xs:
        ys.append (lsf(x))

    plt.plot(xs,c='b')
    plt.plot(ys, c='g')
    plt.plot([0,len(xs)-1], [ys[0], ys[-1]])



def test_fig_3_8():
    """ figure 3.8 in Zarchan, p. 108"""
    lsf = LeastSquaresFilter(0.1, order=1)

    xs = [x+3 + random.randn() for x in np.arange (0,10, 0.1)]
    ys = []
    for x in xs:
        ys.append (lsf(x))

    plt.plot(xs)
    plt.plot(ys)


def test_listing_3_4():
    """ listing 3.4 in Zarchan, p. 117"""

    lsf = LeastSquaresFilter(0.1, order=2)

    xs = [5*x*x -x + 2 + 30*random.randn() for x in np.arange (0,10, 0.1)]
    ys = []
    for x in xs:
        ys.append (lsf(x))

    plt.plot(xs)
    plt.plot(ys)


if __name__ == "__main__":
    #test_listing_3_4()

    #test_second_order()
    #fig_3_8()

    test_lsq()