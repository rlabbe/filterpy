import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from filterpy.kalman import SimplexSigmaPoints, unscented_transform


@pytest.mark.parametrize('container', [list, tuple, np.asarray])
def test_simplex_arraylike_mean(container):
    mean = container([1., 2.])
    covariance = np.array([[4., 1.5], [1.5, 9.]])
    original_covariance = covariance.copy()
    points = SimplexSigmaPoints(2)

    sigmas = points.sigma_points(mean, covariance)
    actual_mean, actual_covariance = unscented_transform(
        sigmas, points.Wm, points.Wc)

    assert_allclose(actual_mean, [1., 2.], atol=1e-14)
    assert_allclose(actual_covariance, covariance, atol=1e-14)
    assert_array_equal(mean, [1., 2.])
    assert_array_equal(covariance, original_covariance)


@pytest.mark.parametrize('mean', [3., [3.], (3.,), np.array([3.])])
def test_simplex_one_dimensional_mean(mean):
    points = SimplexSigmaPoints(1)
    sigmas = points.sigma_points(mean, 4.)
    actual_mean, actual_covariance = unscented_transform(
        sigmas, points.Wm, points.Wc)
    assert_allclose(actual_mean, [3.], atol=1e-14)
    assert_allclose(actual_covariance, [[4.]], atol=1e-14)
