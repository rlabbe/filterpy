import numpy as np
import pytest
from numpy.testing import assert_allclose

from filterpy.kalman import KalmanFilter


@pytest.mark.parametrize('column_vector', [False, True])
@pytest.mark.parametrize('chunk_size', [1, 2, 4])
def test_sequential_update_state_shape(column_vector, chunk_size):
    model = KalmanFilter(dim_x=2, dim_z=4)
    mean = np.array([.3, -.2])
    covariance = np.array([[2., .4], [.4, 1.]])
    model.x = mean[:, None] if column_vector else mean.copy()
    model.P = covariance.copy()
    model.H = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
    model.R = np.diag([.3, .5, .4, .6])
    observation = np.array([2., 3., .5, -1.])

    # Condition the joint Gaussian distribution on all four observations.
    cross_covariance = covariance.dot(model.H.T)
    observation_covariance = model.H.dot(cross_covariance) + model.R
    expected_mean = mean + cross_covariance.dot(np.linalg.solve(
        observation_covariance, observation - model.H.dot(mean)))
    expected_covariance = covariance - cross_covariance.dot(np.linalg.solve(
        observation_covariance, cross_covariance.T))

    for start in range(0, 4, chunk_size):
        model.update_sequential(start, observation[start:start + chunk_size])

    assert_allclose(model.x.reshape(-1), expected_mean, atol=1e-12)
    assert_allclose(model.P, expected_covariance, atol=1e-12)
    assert model.x.shape == ((2, 1) if column_vector else (2,))
    assert_allclose(model.z.reshape(-1).astype(float), observation)
