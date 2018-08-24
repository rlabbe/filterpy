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

from math import sin, cos, radians
import numpy.random as random
import numpy as np
from numpy import array
from numpy.random import randn
import matplotlib.pyplot as plt
from filterpy.kalman import IMMEstimator, KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver


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


def test_imm():
    """ This test is drawn from Crassidis [1], example 4.6.

    ** References**

    [1] Crassidis. "Optimal Estimation of Dynamic Systems", CRC Press,
    Second edition.
    """

    r = 100.
    dt = 1.
    phi_sim = np.array(
        [[1, dt, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, dt],
         [0, 0, 0, 1]])

    gam = np.array([[dt**2/2, 0],
                    [dt, 0],
                    [0, dt**2/2],
                    [0, dt]])

    x = np.array([[2000, 0, 10000, -15.]]).T

    simxs = []
    N = 600
    for i in range(N):
        x = np.dot(phi_sim, x)
        if i >= 400:
            x += np.dot(gam, np.array([[.075, .075]]).T)
        simxs.append(x)
    simxs = np.array(simxs)

    zs = np.zeros((N, 2))
    for i in range(len(zs)):
        zs[i, 0] = simxs[i, 0] + randn()*r
        zs[i, 1] = simxs[i, 2] + randn()*r

    '''
    try:
        # data to test against crassidis' IMM matlab code
        zs_tmp = np.genfromtxt('c:/users/rlabbe/dropbox/Crassidis/mycode/xx.csv', delimiter=',')[:-1]
        zs = zs_tmp
    except:
        pass
    '''
    ca = KalmanFilter(6, 2)
    cano = KalmanFilter(6, 2)
    dt2 = (dt**2)/2
    ca.F = np.array(
        [[1, dt, dt2, 0, 0,  0],
         [0, 1,  dt,  0, 0,  0],
         [0, 0,   1,  0, 0,  0],
         [0, 0,   0,  1, dt, dt2],
         [0, 0,   0,  0,  1, dt],
         [0, 0,   0,  0,  0,  1]])
    cano.F = ca.F.copy()

    ca.x = np.array([[2000., 0, 0, 10000, -15, 0]]).T
    cano.x = ca.x.copy()

    ca.P *= 1.e-12
    cano.P *= 1.e-12
    ca.R *= r**2
    cano.R *= r**2
    cano.Q *= 0
    q = np.array([[.05, .125, 1./6],
                  [.125, 1/3, .5],
                  [1./6, .5, 1.]])*1.e-3

    ca.Q[0:3, 0:3] = q
    ca.Q[3:6, 3:6] = q

    ca.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0]])
    cano.H = ca.H.copy()

    filters = [ca, cano]

    trans = np.array([[0.97, 0.03],
                      [0.03, 0.97]])

    bank = IMMEstimator(filters, (0.5, 0.5), trans)

    # ensure __repr__ doesn't have problems
    str(bank)

    s = Saver(bank)
    ca_s = Saver(ca)
    cano_s = Saver(cano)
    for i, z in enumerate(zs):
        z = np.array([z]).T
        bank.update(z)
        bank.predict()

        s.save()
        ca_s.save()
        cano_s.save()

    if DO_PLOT:
        s.to_array()
        ca_s.to_array()
        cano_s.to_array()

        plt.figure()

        plt.subplot(121)
        plt.plot(s.x[:, 0], s.x[:, 3], 'k')
        #plt.plot(cvxs[:, 0], caxs[:, 3])
        #plt.plot(simxs[:, 0], simxs[:, 2], 'g')
        plt.scatter(zs[:, 0], zs[:, 1], marker='+', alpha=0.2)

        plt.subplot(122)
        plt.plot(s.mu[:, 0])
        plt.plot(s.mu[:, 1])
        plt.ylim(0, 1)
        plt.title('probability ratio p(cv)/p(ca)')

        '''plt.figure()
        plt.plot(cvxs, label='CV')
        plt.plot(caxs, label='CA')
        plt.plot(xs[:, 0], label='GT')
        plt.legend()

        plt.figure()
        plt.plot(xs)
        plt.plot(xs[:, 0])'''


def test_misshapen():

    """Ensure we get a ValueError if the filter banks are not designed
    properly
    """

    ca = KalmanFilter(3, 1)
    cv = KalmanFilter(2, 1)

    trans = np.array([[0.97, 0.03],
                      [0.03, 0.97]])

    try:
        IMMEstimator([ca, cv], (0.5, 0.5), trans)
        assert "IMM should raise ValueError on filter banks with filters of different sizes"
    except ValueError:
        pass

    try:
        IMMEstimator([], (0.5, 0.5), trans)
        assert "Should raise ValueError on empty bank"
    except ValueError:
        pass



if __name__ == '__main__':

    #test_misshapen()
    DO_PLOT = True
    test_imm()
