# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http:\\github.com\rlabbe\filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, RKSSmoother


if __name__ == '__main__':

    f = RKSSmoother (dim_x=2, dim_z=1)
    fk = KalmanFilter(dim_x=2, dim_z=1)

    f.x = np.array([[2.],
                    [0.]])        # initial state (location and velocity)

    f.F = np.array([[1.,1.],
                    [0.,1.]])     # state transition matrix

    f.H = np.array([[1.,0.]])     # Measurement function
    f.P *= .01                     # covariance matrix
    f.R *= 2                      # state uncertainty
    f.Q *= 0.01                 # process uncertainty

    f.P[0,0] = .5
    
    fk.x = f.x.copy()
    fk.F = f.F.copy()
    fk.H = f.H.copy()
    fk.P = f.P.copy()
    fk.R = f.R.copy()
    fk.Q = f.Q.copy()


    zs = [t + random.randn()*2 for t in range (40)]
    
    m = f.smooth(zs, 3)
    ms = [x[0,0] for x in m]
    
    mu, cov = fk.batch_filter (zs)
    mus = [x[0,0] for x in mu]
        


    # plot data
    p1, = plt.plot(zs,'r', alpha=0.5)
    p2, = plt.plot (ms,c='b')
    p3, = plt.plot (mus,c='r')
    p4, = plt.plot ([0,len(zs)],[0,len(zs)], 'g') # perfect result
    plt.legend([p1,p2, p3, p4],
               ["measurement", "RKS", "KF output", "ideal"], 4)


    plt.show()

