import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from filterpy.kalman import KalmanFilter, rts_smoother


@pytest.mark.parametrize('vary_f,vary_q', [
    (True, False), (False, True), (True, True), (False, False)])
@pytest.mark.parametrize('column_vector', [False, True])
def test_standalone_rts_matches_joint_gaussian(vary_f, vary_q, column_vector):
    transitions = np.array([.3, 1.2, -.7, .8, 1.1]) if vary_f else np.ones(5)
    noises = np.array([.4, .2, .8, .3, .6]) if vary_q else np.full(5, .4)
    observations = np.array([1., -.5, .8, 1.2, -.2])
    initial_mean, initial_variance, measurement_variance = .3, 1.4, .7
    Fs = [np.array([[value]]) for value in transitions]
    Qs = [np.array([[value]]) for value in noises]

    model = KalmanFilter(dim_x=1, dim_z=1)
    model.x = (np.array([[initial_mean]]) if column_vector
               else np.array([initial_mean]))
    model.P = np.array([[initial_variance]])
    model.H = np.array([[1.]])
    model.R = np.array([[measurement_variance]])
    means, covariances, _, _ = model.batch_filter(observations, Fs=Fs, Qs=Qs)
    original_means, original_covariances = means.copy(), covariances.copy()

    # Express the entire state trajectory in terms of the independent initial
    # state and process noises, then condition jointly on all observations.
    loadings = np.zeros((5, 6))
    coefficients = np.zeros(6)
    coefficients[0] = 1
    for i, transition in enumerate(transitions):
        coefficients *= transition
        coefficients[i + 1] = 1
        loadings[i] = coefficients
    variances = np.r_[initial_variance, noises]
    prior_covariance = (loadings * variances).dot(loadings.T)
    prior_mean = loadings[:, 0] * initial_mean
    observation_covariance = (prior_covariance
                              + np.eye(5) * measurement_variance)
    expected_mean = prior_mean + prior_covariance.dot(
        np.linalg.solve(observation_covariance, observations - prior_mean))
    expected_covariance = prior_covariance - prior_covariance.dot(
        np.linalg.solve(observation_covariance, prior_covariance))

    smoothed = rts_smoother(means, covariances, Fs, Qs)
    assert_allclose(smoothed[0].reshape(-1), expected_mean, atol=1e-12)
    assert_allclose(smoothed[1][:, 0, 0], np.diag(expected_covariance),
                    atol=1e-12)
    class_smoothed = model.rts_smoother(means, covariances, Fs, Qs)
    for actual, expected in zip(smoothed, class_smoothed):
        assert_allclose(actual, expected, atol=1e-12)
    assert_array_equal(means, original_means)
    assert_array_equal(covariances, original_covariances)
