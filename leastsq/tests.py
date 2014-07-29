# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http:\\github.com\rlabbe\filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from filterpy.leastsq import LeastSquaresFilter
import numpy.random as random
import matplotlib.pyplot as plt

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
    test_listing_3_4()

    test_second_order()
    #fig_3_8()