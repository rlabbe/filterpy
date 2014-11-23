# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 07:43:57 2014

@author: rlabbe
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import array
import matplotlib.pyplot as plt
from filterpy.hinfinity import HInfinityFilter




def test_Hinfinity():
    dt = 0.1
    f = HInfinityFilter(2, 1, 0, gamma=.4)

    f.F = array([[1., dt],
                 [0., 1.]])

    f.H = array([[0., 1.]])
    f.x = array([[0., 0.]]).T
    #f.G = array([[dt**2 / 2, dt]]).T

    f.P = 0.01
    f.W = array([[0.0003, 0.005],
                 [0.0050, 0.100]])/ 1000

    f.V = 0.01
    f.Q = 0.01

    xs = []
    vs = []

    for i in range(1,40):
        f.update (5)
        print(f.x.T)
        xs.append(f.x[0,0])
        vs.append(f.x[1,0])
        f.predict()

    plt.subplot(211)
    plt.plot(xs)
    plt.subplot(212)
    plt.plot(vs)

if __name__ == "__main__":
    test_Hinfinity()