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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np
from filterpy.memory import FadingMemoryFilter
from filterpy.gh import GHKFilter


def dotest_2d_data():
    """ tests having multidimensional data for x"""

    fm = FadingMemoryFilter(x0=np.array([[0.,2.],[0.,0.]]), dt=1, order=1, beta=.6)

    xs = [x for x in range(0,50)]

    for x in xs:
        data = [x+randn()*3, x+2+randn()*3]
        fm.update(data)
        plt.scatter(fm.x[0,0], fm.x[0,1], c = 'r')
        plt.scatter(data[0], data[1], c='b')



def dotest_1d(order, beta):
    fm = FadingMemoryFilter(x0=0, dt=1, order=order, beta=beta)

    xs = [x for x in range(0,50)]

    fxs = []
    for x in xs:
        data = x+randn()*3
        fm.update(data)
        plt.scatter(x, fm.x[0], c = 'r')
        fxs.append(fm.x[0])
        plt.scatter(x,data,c='b')

    plt.plot(fxs, c='r')


def test_ghk_formulation():
    beta = .6

    g = 1-beta**3
    h = 1.5*(1+beta)*(1-beta)**2
    k = 0.5*(1-beta)**3

    f1 = GHKFilter(0,0,0,1, g, h, k)
    f2 = FadingMemoryFilter(x0=0, dt=1, order=2, beta=beta)

    def fx(x):
        return .02*x**2 + 2*x - 3

    for i in range(1,100):
        z = fx(i)
        f1.update(z)
        f2.update(z)

        assert abs(f1.x-f2.x[0]) < 1.e-80



if __name__ == "__main__":
    test_ghk_formulation()
    '''dotest_1d(0, .7)
    dotest_1d(1, .7)
    dotest_1d(2, .7)
    plt.figure(2)
    dotest_2d_data()'''