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

from filterpy.common import kinematic_kf, Saver, inv_diagonal, outer_product_sum

import numpy as np
from filterpy.kalman import (MerweScaledSigmaPoints, UnscentedKalmanFilter,
                             ExtendedKalmanFilter)

def test_kinematic_filter():
    global kf

    # make sure the default matrices are shaped correctly
    for dim_x in range(1,4):
        for order in range (0, 3):
            kf = kinematic_kf(dim=dim_x, order=order)
            kf.predict()
            kf.update(np.zeros((dim_x, 1)))


    # H is tricky, make sure it is shaped and assigned correctly
    kf = kinematic_kf(dim=2, order=2)
    assert kf.x.shape == (6, 1)
    assert kf.F.shape == (6, 6)
    assert kf.P.shape == (6, 6)
    assert kf.Q.shape == (6, 6)
    assert kf.R.shape == (2, 2)
    assert kf.H.shape == (2, 6)

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]], dtype=float)
    assert np.array_equal(H, kf.H)

    kf = kinematic_kf(dim=3, order=2, order_by_dim=False)
    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=float)
    assert np.array_equal(H, kf.H)


def test_saver_UKF():
    def fx(x, dt):
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]], dtype=float)

        return np.dot(F, x)

    def hx(x):
        return np.array([x[0], x[2]])

    dt = 0.1
    points = MerweScaledSigmaPoints(4, .1, 2., -1)
    kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)

    z_std = 0.1
    kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
    kf.x = np.array([-1., 1., -1., 1])
    kf.P *= 1.

    zs = [[i, i] for i in range(40)]
    s = Saver(kf, skip_private=False, skip_callable=False, ignore=['z_mean'])
    for z in zs:
        kf.predict()
        kf.update(z)
        #print(kf.x, kf.log_likelihood, kf.P.diagonal())
        s.save()
    s.to_array()


def test_saver_kf():
    kf = kinematic_kf(3, 2, dt=0.1, dim_z=3)
    s = Saver(kf)

    for i in range(10):
        kf.predict()
        kf.update([i, i, i])
        s.save()

    # this will assert if the KalmanFilter did not properly assert
    s.to_array()
    assert len(s.x) == 10
    assert len(s.y) == 10
    assert len(s.K) == 10
    assert (len(s) == len(s.x))

    # Force an exception to occur my malforming K
    kf = kinematic_kf(3, 2, dt=0.1, dim_z=3)
    kf.K = 0.
    s2 = Saver(kf)

    for i in range(10):
        kf.predict()
        kf.update([i, i, i])
        s2.save()

    # this should raise an exception because K is malformed
    try:
        s2.to_array()
        assert False, "Should not have been able to convert s.K into an array"
    except:
        pass


def test_saver_ekf():
    def HJ(x):
        return np.array([[1, 1]])

    def hx(x):
        return np.array([x[0]])

    kf = ExtendedKalmanFilter(2, 1)
    s = Saver(kf)

    for i in range(3):
        kf.predict()
        kf.update([[i]], HJ, hx)
        s.save()

    # this will assert if the KalmanFilter did not properly assert
    s.to_array()
    assert len(s.x) == 3
    assert len(s.y) == 3
    assert len(s.K) == 3


def test_inv_diagonal():

    for i in range(10000):

        n = np.random.randint(1, 50)
        if i == 0:
            n = 1 # test 1x1 matrix as special case

        S = np.diag(np.random.randn(n))

        assert np.allclose(inv_diagonal(S), np.linalg.inv(S))


def test_save_properties():
    global f, s

    class Foo(object):
        aa = 3

        def __init__(self):
            self.x = 7.
            self.a = None

        @property
        def ll(self):
            self.a = Foo.aa
            Foo.aa += 1
            return self.a

    f = Foo()
    assert f.a is None
    s = Saver(f)
    s.save() # try to trigger property writing to Foo.a

    assert s.a[0] == f.a
    assert s.ll[0] == f.a
    assert f.a == 3

    s.save()
    assert s.a[1] == f.a
    assert s.ll[1] == f.a
    assert f.a == 4


def test_outer_product():
    sigmas = np.random.randn(1000000, 2)
    x = np.random.randn(2)

    P1 = outer_product_sum(sigmas-x)

    P2 = 0
    for s in sigmas:
        y = s - x
        P2 += np.outer(y, y)
    assert np.allclose(P1, P2)





if __name__ == "__main__":
    #test_repeaters()
    '''test_save_properties()

    test_saver_kf()
    test_saver_ekf()
    test_inv_diagonal()

    ITERS = 1000000
    #test_mahalanobis()

    test_kinematic_filter()'''


