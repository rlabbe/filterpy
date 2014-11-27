# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 20:07:18 2014

@author: rlabbe
"""

import math
from numpy import array, dot, asarray
from numpy.random import randn
import matplotlib.pyplot as plt

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import SigmaPoints, ScaledPoints
from filterpy.common import Q_discrete_white_noise
dt = 0.1
y = 20
p_platform=(0,20)

scaled_pts = ScaledPoints(n=2, alpha=1.0001, beta=0., kappa=1)
pts = SigmaPoints(n=2, kappa=0.)
sf = UKF(2, 1, dt, scaled_pts)
sf.Q = Q_discrete_white_noise(2, dt, .1)

f = UKF(2, 1, dt, pts)
f.Q = Q_discrete_white_noise(2, dt, .1)

def fx(x,dt):
    return array([x[0]+x[1], x[1]])

def hx(x):
    return math.atan2(p_platform[1],x[0]-p_platform[0])

xs_scaled = []
xs = []
for i in range(300):

    angle = hx([i+randn()*.1, 0]) + randn()
    sf.predict(fx)
    sf.update(angle, hx)
    xs_scaled.append(sf.x)

    f.predict(fx)
    f.update(angle, hx)
    xs.append(f.x)

xs_scaled = asarray(xs_scaled)
xs = asarray(xs)

plt.subplot(211)
plt.plot(xs_scaled[:,0],label='scaled')
plt.plot(xs[:,0], label='Julier')
plt.legend(loc=4)

plt.subplot(212)
plt.plot(xs_scaled[:,1],label='scaled')
plt.plot(xs[:,1], label='Julier')
plt.legend(loc=4)
plt.show()


