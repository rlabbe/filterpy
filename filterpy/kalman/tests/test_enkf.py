# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 09:10:29 2014

@author: rlabbe
"""

from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.common import Q_discrete_white_noise

#def test_noisy_1d():

def hx(x):
    return np.array([x[0]])

F = np.array([[1.,1],[0,1]])
def fx(x, dt):
    return np.dot(F,x)


f = EnKF(dim_x=2, dim_z=1, dt=.1, N=2,hx=hx, fx=fx)

f.x = np.array([0., 1.])

f.P *= 20.
f.R *= 3.
f.Q = Q_discrete_white_noise(2, .1, 1.1)

f.initialize (f.x, f.P)
print(np.sum(f.sigmas,axis=0)/ f.N)

measurements = []
results = []

zs = []
for t in range (100):
    # create measurement = t plus white noise
    z = t + randn()*3
    zs.append(z)

    # perform kalman filtering
    f.update([z])
    f.predict()

    print(f.P)

    # save data
    results.append (f.x[0])
    measurements.append(z)

plt.plot(results)
plt.plot(measurements)

#test_noisy_1d()