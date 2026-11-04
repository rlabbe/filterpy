from copy import deepcopy
import numpy as np
import pytest
from filterpy.kalman import ExtendedKalmanFilter


def make_filter():
    f = ExtendedKalmanFilter(2, 1)
    f.x = np.array([[1.0], [2.0]])
    f.P = np.array([[2.0, 0.3], [0.3, 1.0]])
    f.F = np.array([[1.0, 0.5], [0.0, 1.0]])
    f.Q = np.eye(2) * 0.2
    f.H = np.array([[1.0, 0.0]])
    f.B = np.array([[0.2], [0.4]])
    return f


@pytest.mark.parametrize("measurement", [None, np.array([[2.5]])])
def test_combined_step_matches_separate_predict_and_update(measurement):
    combined = make_filter()
    separate = deepcopy(combined)
    jacobian = lambda x: np.array([[1.0, 0.0]])
    observation = lambda x: x[:1]
    separate.predict(u=np.array([[0.7]]))
    separate.update(measurement, jacobian, observation)
    combined.predict_update(measurement, jacobian, observation, u=np.array([[0.7]]))
    for name in ["x", "P", "x_prior", "P_prior", "x_post", "P_post"]:
        np.testing.assert_allclose(getattr(combined, name), getattr(separate, name))


def test_missing_measurement_does_not_call_observation_functions():
    f = make_filter()

    def unused(x):
        raise AssertionError("observation evaluated for missing measurement")

    f.predict_update(None, unused, unused)
    np.testing.assert_allclose(f.x, [[2.0], [2.0]])
    assert f.z[0, 0] is None
