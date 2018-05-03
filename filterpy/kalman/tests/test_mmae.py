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

import numpy.random as random
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, MMAEFilterBank
from numpy import array
from filterpy.common import Q_discrete_white_noise, Saver
import matplotlib.pyplot as plt
from numpy.random import randn
from math import sin, cos, radians


DO_PLOT = False



class NoisySensor(object):
    def __init__(self, noise_factor=1):
        self.noise_factor = noise_factor

    def sense(self, pos):
        return (pos[0] + randn()*self.noise_factor,
                pos[1] + randn()*self.noise_factor)



def angle_between(x, y):
  return min(y-x, y-x+360, y-x-360, key=abs)

class ManeuveringTarget(object):
    def __init__(self, x0, y0, v0, heading):
        self.x = x0
        self.y = y0
        self.vel = v0
        self.hdg = heading

        self.cmd_vel = v0
        self.cmd_hdg = heading
        self.vel_step = 0
        self.hdg_step = 0
        self.vel_delta = 0
        self.hdg_delta = 0


    def update(self):
        vx = self.vel * cos(radians(90-self.hdg))
        vy = self.vel * sin(radians(90-self.hdg))
        self.x += vx
        self.y += vy

        if self.hdg_step > 0:
            self.hdg_step -= 1
            self.hdg += self.hdg_delta

        if self.vel_step > 0:
            self.vel_step -= 1
            self.vel += self.vel_delta
        return (self.x, self.y)


    def set_commanded_heading(self, hdg_degrees, steps):
        self.cmd_hdg = hdg_degrees
        self.hdg_delta = angle_between(self.cmd_hdg,
                                       self.hdg) / steps
        if abs(self.hdg_delta) > 0:
            self.hdg_step = steps
        else:
            self.hdg_step = 0


    def set_commanded_speed(self, speed, steps):
        self.cmd_vel = speed
        self.vel_delta = (self.cmd_vel - self.vel) / steps
        if abs(self.vel_delta) > 0:
            self.vel_step = steps
        else:
            self.vel_step = 0


def make_cv_filter(dt, noise_factor):
    cvfilter = KalmanFilter(dim_x = 2, dim_z=1)
    cvfilter.x = array([0., 0.])
    cvfilter.P *= 3
    cvfilter.R *= noise_factor**2
    cvfilter.F = array([[1, dt],
                        [0,  1]], dtype=float)
    cvfilter.H = array([[1, 0]], dtype=float)
    cvfilter.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.02)
    return cvfilter


def make_ca_filter(dt, noise_factor):
    cafilter = KalmanFilter(dim_x=3, dim_z=1)
    cafilter.x = array([0., 0., 0.])
    cafilter.P *= 3
    cafilter.R *= noise_factor**2
    cafilter.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.02)
    cafilter.F = array([[1, dt, 0.5*dt*dt],
                        [0, 1,         dt],
                        [0, 0,          1]], dtype=float)
    cafilter.H = array([[1, 0, 0]], dtype=float)
    return cafilter



def generate_data(steady_count, noise_factor):
    t = ManeuveringTarget(x0=0, y0=0, v0=0.3, heading=0)
    xs = []
    ys = []

    for i in range(30):
        x, y = t.update()
        xs.append(x)
        ys.append(y)

    t.set_commanded_heading(310, 25)
    t.set_commanded_speed(1, 15)

    for i in range(steady_count):
        x, y = t.update()
        xs.append(x)
        ys.append(y)

    ns = NoisySensor(noise_factor=noise_factor)
    pos = array(list(zip(xs, ys)))
    zs = array([ns.sense(p) for p in pos])
    return pos, zs



def test_MMAE2():
    dt = 0.1
    pos, zs = generate_data(120, noise_factor=0.6)
    z_xs = zs[:, 0]

    dt = 0.1
    ca = make_ca_filter(dt, noise_factor=0.6)
    cv = make_ca_filter(dt, noise_factor=0.6)
    cv.F[:, 2] = 0 # remove acceleration term
    cv.P[2, 2] = 0
    cv.Q[2, 2] = 0

    filters = [cv, ca]


    bank = MMAEFilterBank(filters, (0.5, 0.5), dim_x=3, H=ca.H)

    xs, probs = [], []
    cvxs, caxs = [], []
    s = Saver(bank)
    for i, z in enumerate(z_xs):
        bank.predict()
        bank.update(z)
        xs.append(bank.x[0])
        cvxs.append(cv.x[0])
        caxs.append(ca.x[0])
        print(i, cv.likelihood, ca.likelihood, bank.p)
        s.save()
        probs.append(bank.p[0] / bank.p[1])
    s.to_array()

    if DO_PLOT:
        plt.subplot(121)
        plt.plot(xs)
        plt.plot(pos[:, 0])
        plt.subplot(122)
        plt.plot(probs)
        plt.title('probability ratio p(cv)/p(ca)')

        plt.figure()
        plt.plot(cvxs, label='CV')
        plt.plot(caxs, label='CA')
        plt.plot(pos[:, 0])
        plt.legend()

        plt.figure()
        plt.plot(xs)
        plt.plot(pos[:, 0])

    return bank

if __name__ == '__main__':
    DO_PLOT = True
    test_MMAE2()