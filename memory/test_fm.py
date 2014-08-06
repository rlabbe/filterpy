# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 20:47:55 2014

@author: rlabbe
"""

from filterpy.memory import FadingMemoryFilter

import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np


def test_2d_data():
    """ tests having multidimensional data for x"""
    
    fm = FadingMemoryFilter(x0=np.array([[0,2],[0,0]]), dt=1, order=1, beta=.6)

    xs = [x for x in range(0,50)]

    for x in xs:
        data = [x+randn()*3, x+2+randn()*3]
        fm.update(data)
        plt.scatter(fm.x[0,0], fm.x[0,1], c = 'r')
        plt.scatter(data[0], data[1], c='b')

    plt.show()



def test_1d(order, beta):
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
    plt.show()


if __name__ == "__main__":
    test_1d(0, .7)
    test_1d(1, .7)
    test_1d(2, .7)
    plt.figure(2)
    test_2d_data()