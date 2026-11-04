import numpy as np
import pytest
from filterpy.kalman import InformationFilter, KalmanFilter


@pytest.mark.parametrize("dim_x,dim_z", [(2, 1), (2, 2), (2, 3)])
def test_likelihood_matches_covariance_filter(dim_x, dim_z):
    regular = KalmanFilter(dim_x, dim_z)
    info = InformationFilter(dim_x, dim_z)
    regular.x = np.array([[0.4], [-0.2]])
    regular.P = np.array([[2.0, 0.3], [0.3, 1.0]])
    regular.H = np.arange(1, dim_x * dim_z + 1, dtype=float).reshape(dim_z, dim_x) / 4
    regular.R = np.eye(dim_z) * 0.7 + np.ones((dim_z, dim_z)) * 0.1
    info.x, info.H = regular.x.copy(), regular.H.copy()
    info.P_inv, info.R_inv = np.linalg.inv(regular.P), np.linalg.inv(regular.R)
    measurement = np.arange(dim_z, dtype=float).reshape(-1, 1) + 0.5
    regular.update(measurement)
    info.update(measurement)
    np.testing.assert_allclose(info.x, regular.x)
    np.testing.assert_allclose(info.P, regular.P)
    assert info.log_likelihood == pytest.approx(regular.log_likelihood)
    assert info.likelihood == pytest.approx(regular.likelihood)


def test_update_does_not_require_a_covariance_when_likelihood_disabled():
    info = InformationFilter(2, 2, compute_log_likelihood=False)
    info.P_inv = np.zeros((2, 2))
    info.H = np.eye(2)
    info.update(np.array([[2.0], [3.0]]))
    np.testing.assert_allclose(info.x, [[2.0], [3.0]])
