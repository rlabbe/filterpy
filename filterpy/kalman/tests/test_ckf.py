# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 08:03:07 2016

@author: rlabbe
"""

from filterpy.kalman import CubatureKalmanFilter as CKF
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from numpy.random import randn
import pytest
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

    for i in range(50):
        z = np.array([[i+randn()*0.1]])
        ckf.predict()
        ckf.update(z)
        kf.predict()
        kf.update(z[0])
        assert abs(ckf.x[0] - kf.x[0]) < 1e-10
        assert abs(ckf.x[1] - kf.x[1]) < 1e-10

        # test mahalanobis
        a = np.zeros(kf.y.shape)
        maha = scipy_mahalanobis(a, kf.y, kf.SI)
        assert kf.mahalanobis == approx(maha)


def _run_linear_ckf_history():
    def fx(x, dt):
        F = np.array([[1., dt],
                      [0., 1.]])
        return np.dot(F, x)

    def hx(x):
        return x[0:1]

    dt = 0.1
    sig_t = 0.1
    sig_o = 0.2

    np.random.seed(100)
    zs = [i + 1 + np.random.normal(scale=sig_o) for i in range(50)]

    ckf = CKF(dim_x=2, dim_z=1, dt=dt, hx=hx, fx=fx)
    ckf.x = np.array([[0.], [1.]])
    ckf.R = np.array([[sig_o**2]])
    ckf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=sig_t**2)

    xs, ps = [], []
    for z in zs:
        ckf.predict()
        ckf.update(np.array([[z]]))
        xs.append(ckf.x.copy())
        ps.append(ckf.P.copy())

    return zs, ckf, np.asarray(xs), np.asarray(ps)


def test_ckf_rts_smoother_linear_matches_kf():
    def fx(x, dt):
        F = np.array([[1., dt],
                      [0., 1.]])
        return np.dot(F, x)

    H = np.array([[1., 0.]])
    dt = 0.1
    sig_t = 0.1
    sig_o = 0.2

    zs, ckf, xs, ps = _run_linear_ckf_history()
    x_ckf, p_ckf, k_ckf = ckf.rts_smoother(xs, ps)

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [1.]])
    kf.F = np.array([[1., dt],
                     [0., 1.]])
    kf.H = H
    kf.R = np.array([[sig_o**2]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=sig_t**2)

    mu_kf, cov_kf, _, _ = kf.batch_filter(zs)
    x_kf, p_kf, k_kf, _ = kf.rts_smoother(mu_kf, cov_kf)

    assert np.allclose(x_ckf[:, 0, 0], x_kf[:, 0, 0], atol=1e-4)
    assert np.allclose(x_ckf[:, 1, 0], x_kf[:, 1, 0], atol=1e-4)
    assert np.allclose(p_ckf, p_kf, atol=1e-5)
    assert np.allclose(k_ckf[:-1], k_kf[:-1], atol=2e-4)


def test_ckf_rts_smoother_flat_and_column_histories_match():
    _, ckf, xs, ps = _run_linear_ckf_history()

    x_col, p_col, k_col = ckf.rts_smoother(xs, ps)
    x_flat, p_flat, k_flat = ckf.rts_smoother(xs[:, :, 0], ps)
    x_q, p_q, k_q = ckf.rts_smoother(
        xs,
        ps,
        Qs=[ckf.Q] * len(xs),
        dts=[ckf._dt] * len(xs),
    )

    assert np.allclose(x_col[:, :, 0], x_flat)
    assert np.allclose(p_col, p_flat)
    assert np.allclose(k_col, k_flat)
    assert np.allclose(x_col, x_q)
    assert np.allclose(p_col, p_q)
    assert np.allclose(k_col, k_q)


def test_ckf_rts_smoother_length_mismatch():
    _, ckf, xs, ps = _run_linear_ckf_history()

    with pytest.raises(ValueError, match="same length"):
        ckf.rts_smoother(xs[:-1], ps)


if __name__ == "__main__":
    test_1d()
