# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 08:03:07 2016

@author: rlabbe
"""

from filterpy.common import Saver
from filterpy.kalman import CubatureKalmanFilter as CKF
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
from numpy.random import randn
from pytest import approx
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis


def test_1d():
    def fx(x, dt):
        F = np.array([[1., dt],
                      [0,  1]])

        return np.dot(F, x)

    def hx(x):
        return x[0:1]

    ckf = CKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)

    ckf.x = np.array([[1.], [2.]])
    ckf.P = np.array([[1, 1.1],
                      [1.1, 3]])

    ckf.R = np.eye(1) * .05
    ckf.Q = np.array([[0., 0], [0., .001]])

    dt = 0.1
    points = MerweScaledSigmaPoints(2, .1, 2., -1)
    kf = UKF(dim_x=2, dim_z=1, dt=dt, fx=fx, hx=hx, points=points)

    kf.x = np.array([1, 2])
    kf.P = np.array([[1, 1.1],
                     [1.1, 3]])
    kf.R *= 0.05
    kf.Q = np.array([[0., 0], [0., .001]])

    s = Saver(kf)
    for i in range(50):
        z = np.array([[i+randn()*0.1]])
        ckf.predict()
        ckf.update(z)
        kf.predict()
        kf.update(z[0])
        assert abs(ckf.x[0] - kf.x[0]) < 1e-10
        assert abs(ckf.x[1] - kf.x[1]) < 1e-10
        s.save()

        # test mahalanobis
        a = np.zeros(kf.y.shape)
        maha = scipy_mahalanobis(a, kf.y, kf.SI)
        assert kf.mahalanobis == approx(maha)

    s.to_array()


if __name__ == "__main__":
    test_1d()
