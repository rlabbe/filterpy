import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from filterpy.kalman import IMMEstimator, KalmanFilter


@pytest.mark.parametrize('column_vector', [False, True])
@pytest.mark.parametrize('transition', [
    np.array([[.1, .9], [.2, .8]]),
    np.array([[.8, .2], [.9, .1]]),
    np.eye(2)])
def test_imm_prior_matches_enumerated_modes(transition, column_vector):
    means, variances = np.array([0., 10.]), np.array([1., 4.])
    dynamics, noises = np.array([1., 2.]), np.array([.2, .7])
    probabilities = np.array([.8, .2])
    filters = []
    for i in range(2):
        model = KalmanFilter(dim_x=1, dim_z=1)
        model.x = (np.array([[means[i]]]) if column_vector
                   else np.array([means[i]]))
        model.P = np.array([[variances[i]]])
        model.F = np.array([[dynamics[i]]])
        model.Q = np.array([[noises[i]]])
        model.H = np.array([[1.]])
        model.R = np.array([[.5]])
        filters.append(model)

    # Enumerate all previous/current mode pairs rather than using IMM helpers.
    weights = probabilities[:, None] * transition
    branch_means = means[:, None] * dynamics[None, :]
    branch_variances = variances[:, None] * dynamics[None, :]**2 + noises
    expected_mean = np.sum(weights * branch_means)
    expected_variance = np.sum(weights * (
        branch_variances + (branch_means - expected_mean)**2))

    estimator = IMMEstimator(filters, probabilities, transition)
    estimator.predict()
    assert_allclose(estimator.x.reshape(-1), [expected_mean])
    assert_allclose(estimator.P, [[expected_variance]])
    assert_array_equal(estimator.x_prior, estimator.x)
    assert_array_equal(estimator.P_prior, estimator.P)
    assert_array_equal(estimator.mu, probabilities)

    # Measurement correction still uses posterior, not prior, mode weights.
    observation = 3.
    prior_mean = np.array([f.x.item() for f in filters])
    prior_variance = np.array([f.P.item() for f in filters])
    observation_variance = prior_variance + .5
    likelihood = norm.pdf(observation, prior_mean,
                          np.sqrt(observation_variance))
    posterior_weights = weights.sum(axis=0) * likelihood
    posterior_weights /= posterior_weights.sum()
    gain = prior_variance / observation_variance
    posterior_means = prior_mean + gain * (observation - prior_mean)
    posterior_variances = (1 - gain) * prior_variance
    posterior_mean = np.dot(posterior_weights, posterior_means)
    posterior_variance = np.dot(
        posterior_weights,
        posterior_variances + (posterior_means - posterior_mean)**2)
    estimator.update(np.array([observation]))
    assert_allclose(estimator.mu, posterior_weights)
    assert_allclose(estimator.x.reshape(-1), [posterior_mean])
    assert_allclose(estimator.P, [[posterior_variance]])
