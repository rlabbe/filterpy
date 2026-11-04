import numpy as np
from filterpy.kalman import KalmanFilter, SquareRootKalmanFilter


def test_saved_posterior_matches_covariance_filter_and_survives_prediction():
    regular, square = KalmanFilter(2, 1), SquareRootKalmanFilter(2, 1)
    for f in (regular, square):
        f.x = np.array([[0.0], [1.0]])
        f.P = np.array([[3.0, 0.4], [0.4, 2.0]])
        f.F = np.array([[1.0, 0.5], [0.0, 1.0]])
        f.H = np.array([[1.0, 0.0]])
        f.Q = np.array([[0.2, 0.05], [0.05, 0.1]])
        f.R = np.array([[0.7]])
    for f in (regular, square):
        f.predict()
        f.update(np.array([[1.2]]))
    np.testing.assert_allclose(square.P_post, regular.P_post, atol=1e-14)
    posterior = regular.P_post.copy()
    square.predict()
    np.testing.assert_allclose(square.P_post, posterior, atol=1e-14)
    assert not np.allclose(square.P_prior, posterior)
    square.update(None)
    np.testing.assert_allclose(square.P_post, square.P)
