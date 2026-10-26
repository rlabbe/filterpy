import random
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.stats import t

import filterpy.stats.stats as stats


@pytest.mark.parametrize('df', [3, 7, 30])
def test_student_t_uses_chi_squared_scale(monkeypatch, df):
    generator = Mock()
    generator.gauss.return_value = 4.0
    generator.gammavariate.return_value = float(df)
    monkeypatch.setattr(stats, 'random', generator)

    # A chi-squared draw equal to df makes the t denominator exactly one.
    assert stats.rand_student_t(df, mu=1, std=2) == 5.0
    generator.gauss.assert_called_once_with(0, 2)
    generator.gammavariate.assert_called_once_with(0.5 * df, 2.0)


@pytest.mark.parametrize('df,mu,scale', [(7, 3, 2), (30, -2, .5), (7, 0, 1)])
def test_student_t_variance_matches_scipy(monkeypatch, df, mu, scale):
    monkeypatch.setattr(stats, 'random', random.Random(1846))
    samples = [stats.rand_student_t(df, mu=mu, std=scale)
               for _ in range(40000)]
    # SciPy's independent Student-t distribution supplies the variance.
    expected_variance = t.var(df, scale=scale)
    assert np.var(samples) == pytest.approx(expected_variance, rel=0.04)
    assert np.mean(samples) == pytest.approx(mu, abs=.02 * scale)
