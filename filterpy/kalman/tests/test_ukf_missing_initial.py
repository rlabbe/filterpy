import numpy as np
import pytest

from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


def make_filter(dimension):
    points = MerweScaledSigmaPoints(dimension, alpha=0.3, beta=2, kappa=0)
    ukf = UnscentedKalmanFilter(
        dimension, dimension, dt=1,
        hx=lambda x: x,
        fx=lambda x, dt: x + dt,
        points=points,
    )
    ukf.Q *= 0.1
    ukf.R *= 0.2
    return ukf


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("all_missing", [False, True])
def test_batch_filter_initial_missing_measurements(dimension, all_missing):
    measurements = [None, None, None]
    if not all_missing:
        measurements[2] = np.full(dimension, 4.0)
    batch = make_filter(dimension)
    reference = make_filter(dimension)
    means, covariances = batch.batch_filter(measurements)
    for i, measurement in enumerate(measurements):
        reference.predict()
        reference.update(measurement)
        np.testing.assert_allclose(means[i], reference.x)
        np.testing.assert_allclose(covariances[i], reference.P)


def test_initial_missing_measurement_still_checks_observed_dimension():
    ukf = make_filter(2)
    with pytest.raises(TypeError, match="length 2"):
        ukf.batch_filter([None, np.array([1.0])])
