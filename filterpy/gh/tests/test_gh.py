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

from filterpy.common import Saver
from filterpy.gh import (GHFilter, GHKFilter, least_squares_parameters,
                         optimal_noise_smoothing, GHFilterOrder)
from numpy import array
from numpy.random import randn
import matplotlib.pyplot as plt


def test_least_squares():

    """ there is an alternative form for computing h for the least squares.
    It works for all but the very first term (n=0). Use it to partially test
    the output of least_squares_parameters(). This test does not test that
    g is correct"""

    for n in range (1, 100):
        g,h = least_squares_parameters(n)

        h2 = 4 - 2*g - (4*(g-2)**2 - 3*g**2)**.5

        assert abs(h2-h) < 1.e-12


def test_1d_array():
    f1 = GHFilter (0, 0, 1, .8, .2)
    f2 = GHFilter (array([0]), array([0]), 1, .8, .2)

    str(f1)
    str(f2)

    # test both give same answers, and that we can
    # use a scalar for the measurment
    for i in range(1,10):
        f1.update(i)
        f2.update(i)

        assert f1.x == f2.x[0]
        assert f1.dx == f2.dx[0]

        assert f1.VRF() == f2.VRF()

    # test using an array for the measurement
    s1 = Saver(f1)
    s2 = Saver(f2)

    for i in range(1,10):
        f1.update(i)
        f2.update(array([i]))

        s1.save()
        s2.save()

        assert f1.x == f2.x[0]
        assert f1.dx == f2.dx[0]

        assert f1.VRF() == f2.VRF()
    s1.to_array()
    s2.to_array()


def test_2d_array():
    """ test using 2 independent variables for the
    state variable.
    """

    f = GHFilter(array([0,1]), array([0,0]), 1, .8, .2)
    f0 = GHFilter(0, 0, 1, .8, .2)
    f1 = GHFilter(1, 0, 1, .8, .2)

    # test using scalar in update (not normal, but possible)
    for i in range (1,10):
        f.update (i)
        f0.update(i)
        f1.update(i)

        assert f.x[0] == f0.x
        assert f.x[1] == f1.x

        assert f.dx[0] == f0.dx
        assert f.dx[1] == f1.dx

    # test using array for update (typical scenario)
    f = GHFilter(array([0,1]), array([0,0]), 1, .8, .2)
    f0 = GHFilter(0, 0, 1, .8, .2)
    f1 = GHFilter(1, 0, 1, .8, .2)

    for i in range (1,10):
        f.update (array([i, i+3]))
        f0.update(i)
        f1.update(i+3)

        assert f.x[0] == f0.x
        assert f.x[1] == f1.x

        assert f.dx[0] == f0.dx
        assert f.dx[1] == f1.dx

        assert f.VRF() == f0.VRF()
        assert f.VRF() == f1.VRF()



def optimal_test():
    def fx(x):
        return .1*x**2 + 3*x -4

    g,h,k = optimal_noise_smoothing(.2)
    f = GHKFilter(-4,0,0,1,g,h,k)

    ys = []
    zs = []
    for i in range(100):
        z = fx(i) + randn()*10
        f.update(z)
        ys.append(f.x)
        zs.append(z)

    plt.plot(ys)
    plt.plot(zs)


def test_GHFilterOrder():
    def fx(x):
        return 2*x+1

    f1 = GHFilterOrder(x0=array([0,0]), dt=1, order=1, g=.6, h=.02)
    f2 = GHFilter(x=0, dx=0, dt=1, g=.6, h=.02)

    for i in range(100):
        z = fx(i) + randn()
        f1.update(z)
        f2.update(z)

        assert abs(f1.x[0]-f2.x) < 1.e-18


if __name__ == "__main__":
    optimal_test()

    test_least_squares()
    test_1d_array()
    test_2d_array()

    test_GHFilterOrder()

    print('all passed')
