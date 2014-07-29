# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 09:33:19 2014

@author: rlabbe
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, RKSSmoother


if __name__ == '__main__':

    f = KalmanFilter (dim_x=2, dim_z=1)

    f.x = np.array([[2.],
                    [0.]])        # initial state (location and velocity)

    f.F = np.array([[1.,1.],
                    [0.,1.]])     # state transition matrix

    f.H = np.array([[1.,0.]])     # Measurement function
    f.P *= .01                     # covariance matrix
    f.R *= 5                      # state uncertainty
    f.Q *= 0.0001                 # process uncertainty

    f.P[0,0] = 1

    zs = [t + random.randn()*20 for t in range (100)]
    m,c = f.batch_filter(zs)

    smoother = RKSSmoother()
    m2, c2, d = smoother.smooth(m, c, f.F, f.Q)
    for i in range(0):
        m2[i]=m2[-1]

    # plot data
    p1, = plt.plot(zs,'r', alpha=0.5)
    p2, = plt.plot (m[:,0],'b')
    p4, = plt.plot(m2[:,0], 'm')
    p3, = plt.plot ([0,100],[0,100], 'g') # perfect result
    plt.legend([p1,p2, p3, p4],
               ["noisy measurement", "KF output", "ideal", "smooth"], 4)


    plt.show()

