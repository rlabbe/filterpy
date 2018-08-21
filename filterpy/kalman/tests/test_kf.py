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

from __future__ import absolute_import, division, print_function


import numpy.random as random
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from pytest import approx
from filterpy.kalman import KalmanFilter, update, predict, batch_filter
from filterpy.common import Q_discrete_white_noise, kinematic_kf, Saver
from scipy.linalg import block_diag, norm
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

DO_PLOT = False


class PosSensor1(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]

    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        return [self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std]


def const_vel_filter(dt, x0=0, x_ndim=1, P_diag=(1., 1.), R_std=1.,
                     Q_var=0.0001):
    """ helper, constructs 1d, constant velocity filter"""
    f = KalmanFilter(dim_x=2, dim_z=1)

    if x_ndim == 1:
        f.x = np.array([x0, 0.])
    else:
        f.x = np.array([[x0, 0.]]).T

    f.F = np.array([[1., dt],
                    [0., 1.]])

    f.H = np.array([[1., 0.]])
    f.P = np.diag(P_diag)
    f.R = np.eye(1) * (R_std**2)
    f.Q = Q_discrete_white_noise(2, dt, Q_var)

    return f


def const_vel_filter_2d(dt, x_ndim=1, P_diag=(1., 1, 1, 1), R_std=1.,
                        Q_var=0.0001):
    """ helper, constructs 1d, constant velocity filter"""

    kf = KalmanFilter(dim_x=4, dim_z=2)

    kf.x = np.array([[0., 0., 0., 0.]]).T
    kf.P *= np.diag(P_diag)
    kf.F = np.array([[1., dt, 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., dt],
                     [0., 0., 0., 1.]])

    kf.H = np.array([[1., 0, 0, 0],
                     [0., 0, 1, 0]])

    kf.R *= np.eye(2) * (R_std**2)
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)
    kf.Q = block_diag(q, q)

    return kf


def test_noisy_1d():
    f = KalmanFilter(dim_x=2, dim_z=1)

    f.x = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    f.F = np.array([[1., 1.],
                    [0., 1.]])    # state transition matrix

    f.H = np.array([[1., 0.]])    # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R = 5                       # state uncertainty
    f.Q = 0.0001                  # process uncertainty

    measurements = []
    results = []

    zs = []
    for t in range(100):
        # create measurement = t plus white noise
        z = t + random.randn()*20
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        # save data
        results.append(f.x[0, 0])
        measurements.append(z)

        # test mahalanobis
        a = np.zeros(f.y.shape)
        maha = scipy_mahalanobis(a, f.y, f.SI)
        assert f.mahalanobis == approx(maha)


    # now do a batch run with the stored z values so we can test that
    # it is working the same as the recursive implementation.
    # give slightly different P so result is slightly different
    f.x = np.array([[2., 0]]).T
    f.P = np.eye(2) * 100.
    s = Saver(f)
    m, c, _, _ = f.batch_filter(zs, update_first=False, saver=s)
    s.to_array()
    assert len(s.x) == len(zs)
    assert len(s.x) == len(s)

    # plot data
    if DO_PLOT:
        p1, = plt.plot(measurements, 'r', alpha=0.5)
        p2, = plt.plot(results, 'b')
        p4, = plt.plot(m[:, 0], 'm')
        p3, = plt.plot([0, 100], [0, 100], 'g')  # perfect result
        plt.legend([p1, p2, p3, p4],
                   ["noisy measurement", "KF output", "ideal", "batch"], loc=4)
        plt.show()


def test_1d_vel():
    from scipy.linalg import inv
    from numpy import dot
    global ks
    dt = 1.
    std_z = 0.0001

    x = np.array([[0.], [0.]])

    F = np.array([[1., dt],
                  [0., 1.]])

    H = np.array([[1., 0.]])
    P = np.eye(2)
    R = np.eye(1)*std_z**2
    Q = np.eye(2)*0.001

    measurements = []

    xest = []
    ks = []
    pos = 0.
    for t in range(20):
        z = pos + random.randn() * std_z
        pos += 100

        # perform kalman filtering
        x = dot(F, x)
        P = dot(dot(F, P), F.T) + Q

        P2 = P.copy()
        P2[0, 1] = 0  # force there to be no correlation
        P2[1, 0] = 0
        S = dot(dot(H, P2), H.T) + R
        K = dot(dot(P2, H.T), inv(S))
        y = z - dot(H, x)
        x = x + dot(K, y)

        # save data
        xest.append(x.copy())
        measurements.append(z)
        ks.append(K.copy())

    xest = np.array(xest)
    ks = np.array(ks)
    # plot data
    if DO_PLOT:
        plt.subplot(121)
        plt.plot(xest[:, 1])
        plt.subplot(122)
        plt.plot(ks[:, 1])
        plt.show()


def test_noisy_11d():
    f = KalmanFilter(dim_x=2, dim_z=1)

    f.x = np.array([2., 0])      # initial state (location and velocity)

    f.F = np.array([[1., 1.],
                    [0., 1.]])    # state transition matrix

    f.H = np.array([[1., 0.]])    # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R = 5                       # state uncertainty
    f.Q = 0.0001                  # process uncertainty

    measurements = []
    results = []

    zs = []
    for t in range(100):
        # create measurement = t plus white noise
        z = t + random.randn()*20
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        # save data
        results.append(f.x[0])
        measurements.append(z)

        # test mahalanobis
        a = np.zeros(f.y.shape)
        maha = scipy_mahalanobis(a, f.y, f.SI)
        assert f.mahalanobis == approx(maha)

    # now do a batch run with the stored z values so we can test that
    # it is working the same as the recursive implementation.
    # give slightly different P so result is slightly different
    f.x = np.array([[2., 0]]).T
    f.P = np.eye(2) * 100.
    m, c, _, _ = f.batch_filter(zs, update_first=False)

    # plot data
    if DO_PLOT:
        p1, = plt.plot(measurements, 'r', alpha=0.5)
        p2, = plt.plot(results, 'b')
        p4, = plt.plot(m[:, 0], 'm')
        p3, = plt.plot([0, 100], [0, 100], 'g')  # perfect result
        plt.legend([p1, p2, p3, p4],
                   ["noisy measurement", "KF output", "ideal", "batch"], loc=4)

        plt.show()


def test_batch_filter():
    f = KalmanFilter(dim_x=2, dim_z=1)

    f.x = np.array([2., 0])      # initial state (location and velocity)

    f.F = np.array([[1., 1.],
                    [0., 1.]])    # state transition matrix

    f.H = np.array([[1., 0.]])    # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R = 5                       # state uncertainty
    f.Q = 0.0001                  # process uncertainty

    zs = [None, 1., 2.]
    m, c, _, _ = f.batch_filter(zs, update_first=False)
    m, c, _, _ = f.batch_filter(zs, update_first=True)


def test_univariate():
    f = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
    f.x = np.array([[0]])
    f.P *= 50
    f.H = np.array([[1.]])
    f.F = np.array([[1.]])
    f.B = np.array([[1.]])
    f.Q = .02
    f.R *= .1

    for i in range(50):
        f.predict()
        f.update(i)


def test_procedure_form():

    dt = 1.
    std_z = 10.1

    x = np.array([[0.], [0.]])
    F = np.array([[1., dt], [0., 1.]])
    H = np.array([[1., 0.]])
    P = np.eye(2)
    R = np.eye(1)*std_z**2
    Q = Q_discrete_white_noise(2, dt, 5.1)

    kf = KalmanFilter(2, 1)
    kf.x = x.copy()
    kf.F = F.copy()
    kf.H = H.copy()
    kf.P = P.copy()
    kf.R = R.copy()
    kf.Q = Q.copy()

    measurements = []
    xest = []
    pos = 0.
    for t in range(2000):
        z = pos + random.randn() * std_z
        pos += 100

        # perform kalman filtering
        x, P = predict(x, P, F, Q)
        kf.predict()
        assert norm(x - kf.x) < 1.e-12
        x, P, _, _, _, _ = update(x, P, z, R, H, True)
        kf.update(z)
        assert norm(x - kf.x) < 1.e-12

        # save data
        xest.append(x.copy())
        measurements.append(z)

    xest = np.asarray(xest)
    measurements = np.asarray(measurements)
    # plot data
    if DO_PLOT:
        plt.plot(xest[:, 0])
        plt.plot(xest[:, 1])
        plt.plot(measurements)


def test_steadystate():

    dim = 7

    cv = kinematic_kf(dim=dim, order=5)

    cv.x[1] = 1.0

    for i in range(100):
        cv.predict()
        cv.update([i]*dim)

    for i in range(100):
        cv.predict_steadystate()
        cv.update_steadystate([i]*dim)
        # test mahalanobis
        a = np.zeros(cv.y.shape)
        maha = scipy_mahalanobis(a, cv.y, cv.SI)
        assert cv.mahalanobis == approx(maha)


def test_procedural_batch_filter():
    f = KalmanFilter(dim_x=2, dim_z=1)

    f.x = np.array([2., 0])

    f.F = np.array([[1., 1.],
                    [0., 1.]])

    f.H = np.array([[1., 0.]])
    f.P = np.eye(2) * 1000.
    f.R = np.eye(1) * 5
    f.Q = Q_discrete_white_noise(2, 1., 0.0001)

    f.test_matrix_dimensions()

    x = np.array([2., 0])

    F = np.array([[1., 1.],
                  [0., 1.]])

    H = np.array([[1., 0.]])
    P = np.eye(2) * 1000.
    R = np.eye(1) * 5
    Q = Q_discrete_white_noise(2, 1., 0.0001)

    zs = [13., None, 1., 2.] * 10
    m, c, _, _ = f.batch_filter(zs, update_first=False)

    n = len(zs)
    mp, cp, _, _ = batch_filter(x, P, zs, [F]*n, [Q]*n, [H]*n, [R]*n)

    for x1, x2 in zip(m, mp):
        assert np.allclose(x1, x2)

    for p1, p2 in zip(c, cp):
        assert np.allclose(p1, p2)


def proc_form():
    """ This is for me to run against the class_form() function to see which,
    if either, runs faster. They within a few ms of each other on my machine
    with Python 3.5.1"""

    dt = 1.
    std_z = 10.1

    x = np.array([[0.], [0.]])
    F = np.array([[1., dt], [0., 1.]])
    H = np.array([[1., 0.]])
    P = np.eye(2)
    R = np.eye(1)*std_z**2
    Q = Q_discrete_white_noise(2, dt, 5.1)

    pos = 0.
    for t in range(2000):
        z = pos + random.randn() * std_z
        pos += 100

        # perform kalman filtering
        x, P = predict(x, P, F, Q)
        x, P, _ = update(z, R, x, P, H)


def class_form():

    dt = 1.
    std_z = 10.1

    f = const_vel_filter(dt, x0=2, R_std=std_z, Q_std=5.1)

    pos = 0.
    for t in range(2000):
        z = pos + random.randn() * std_z
        pos += 100

        f.predict()
        f.update(z)


def test_z_dim():
    f = const_vel_filter(1.0, x0=2, R_std=1., Q_var=5.1)
    f.test_matrix_dimensions()
    f.update(3.)
    assert f.x.shape == (2,)

    f.update([3])
    assert f.x.shape == (2,)

    f.update(np.array([[3]]))
    assert f.x.shape == (2,)

    try:
        f.update(np.array([[[3]]]))
        assert False, "filter should have asserted that [[[3]]] is not a valid form for z"
    except:
        pass

    f = const_vel_filter_2d(1.0, R_std=1., Q_var=5.1)
    try:
        f.update(3)
        assert False, "filter should have asserted that 3 is not a valid form for z"
    except:
        pass

    try:
        f.update([3])
        assert False, "filter should have asserted that [3] is not a valid form for z"
    except:
        pass

    try:
        f.update([3, 3])
        assert False, "filter should have asserted that [3] is not a valid form for z"
    except:
        pass

    try:
        f.update([[3, 3]])
        assert False, "filter should have asserted that [3] is not a valid form for z"
    except:
        pass

    f = const_vel_filter_2d(1.0, R_std=1., Q_var=5.1)
    f.update([[3], [3]])
    f.update(np.array([[3], [3]]))


    # now make sure test_matrix_dimensions() is working

    f.test_matrix_dimensions()
    try:
        f.H = 3
        f.test_matrix_dimensions()
        assert False, "test_matrix_dimensions should have asserted on shape of H"
    except:
        pass

    f = const_vel_filter_2d(1.0, R_std=1., Q_var=5.1)
    try:
        f.R = 3
        f.test_matrix_dimensions()
        assert False, "test_matrix_dimensions should have asserted on shape of R"
    except:
        pass

    try:
        f.R = [3]
        f.test_matrix_dimensions()
        assert False, "test_matrix_dimensions should have asserted on shape of R"
    except:
        pass

    try:
        f.R = [3, 4.]
        f.test_matrix_dimensions()
        assert False, "test_matrix_dimensions should have asserted on shape of R"
    except:
        pass

    f.R = np.diag([3, 4.])
    f.test_matrix_dimensions()


    f = const_vel_filter(1.0, x0=2, R_std=1., Q_var=5.1)

    #test case where x is 1d array
    f.update([[3]])
    f.test_matrix_dimensions(z=3.)
    f.test_matrix_dimensions(z=[3.])

    # test case whre x is 2d array
    f.x = np.array([[0., 0.]]).T
    f.update([[3]])
    f.test_matrix_dimensions(z=3.)
    f.test_matrix_dimensions(z=[3.])

    try:
        f.test_matrix_dimensions(z=[[3.]])
        assert False, "test_matrix_dimensions should have asserted on shape of z"
    except:
        pass

    f = const_vel_filter_2d(1.0, R_std=1., Q_var=5.1)

    # test for 1D value for x, then set to a 2D vector and try again
    for i in range(2):
        try:
            f.test_matrix_dimensions(z=3.)
            assert False, "test_matrix_dimensions should have asserted on shape of z"
        except:
            pass

        try:
            f.test_matrix_dimensions(z=[3.])
            assert False, "test_matrix_dimensions should have asserted on shape of z"
        except:
            pass

        try:
            f.test_matrix_dimensions(z=[3., 3.])
            assert False, "test_matrix_dimensions should have asserted on shape of z"
        except:
            pass
        f.test_matrix_dimensions(z=[[3.], [3.]])
        f.x = np.array([[1, 2, 3, 4.]]).T


def test_default_dims():
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.predict()
    kf.update(np.array([[1.]]).T)


def test_functions():

    x, P = predict(x=10., P=3., u=1., Q=2.**2)
    x, P = update(x=x, P=P, z=12., R=3.5**2)

    x, P = predict(x=np.array([10.]), P=np.array([[3.]]), Q=2.**2)
    x, P = update(x=x, P=P, z=12., H=np.array([[1.]]), R=np.array([[3.5**2]]))

    x = np.array([1., 0])
    P = np.diag([1., 1])
    Q = np.diag([0., 0])
    H = np.array([[1., 0]])

    x, P = predict(x=x, P=P, Q=Q)

    assert x.shape == (2,)
    assert P.shape == (2, 2)

    x, P = update(x, P, z=[1], R=np.array([[1.]]), H=H)

    assert x[0] == 1 and x[1] == 0

    # test velocity predictions
    x, P = predict(x=x, P=P, Q=Q)
    assert x[0] == 1 and x[1] == 0

    x[1] = 1.
    F = np.array([[1., 1], [0, 1]])

    x, P = predict(x=x, F=F, P=P, Q=Q)
    assert x[0] == 2 and x[1] == 1

    x, P = predict(x=x, F=F, P=P, Q=Q)
    assert x[0] == 3 and x[1] == 1


def test_z_checks():
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.update(3.)
    kf.update([3])
    kf.update((3))
    kf.update([[3]])
    kf.update(np.array([[3]]))

    try:
        kf.update([[3, 3]])
        assert False, "accepted bad z shape"
    except ValueError:
        pass

    kf = KalmanFilter(dim_x=3, dim_z=2)
    kf.update([3, 4])
    kf.update([[3, 4]])

    kf.update(np.array([[3, 4]]))
    kf.update(np.array([[3, 4]]).T)


if __name__ == "__main__":
    DO_PLOT = True
    test_functions()
    test_default_dims()
    test_z_checks()
    test_z_dim()
    test_batch_filter()
    test_procedural_batch_filter()

    test_univariate()
    test_noisy_1d()
    test_noisy_11d()
