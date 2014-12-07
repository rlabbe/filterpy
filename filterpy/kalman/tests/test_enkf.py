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

F = np.array([[1., 1.],[0., 1.]])
def fx(x, dt):
    return np.array([x[0]+x[1], x[1]])

x = np.array([0., 1.])
P = np.eye(2)* 100.
f = EnKF(x=x, P=P, dim_z=1, dt=1., N=8, hx=hx, fx=fx)


std_noise = 3.

f.R *= std_noise**2
f.Q = Q_discrete_white_noise(2, 1., 0.01)

f.initialize (f.x, f.P)
print(np.mean(f.sigmas,axis=0))

measurements = []
results = []

zs = []
for t in range (1,100):
    # create measurement = t plus white noise
    z = t + randn()*std_noise
    zs.append(z)

    # perform kalman filtering
    f.predict()
    f.update(np.asarray([z]))

    # save data
    results.append (f.x[0])
    measurements.append(z)

plt.plot(results)
plt.plot(measurements, c='r')

#test_noisy_1d()