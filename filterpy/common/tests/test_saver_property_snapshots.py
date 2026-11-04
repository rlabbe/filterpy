import numpy as np
from filterpy.common import Saver


class FilterWithProperties:
    def __init__(self):
        self._state = np.array([1.0, 2.0])
        self._metadata = {"history": [1]}

    @property
    def state(self):
        return self._state

    @property
    def metadata(self):
        return self._metadata


def test_property_snapshots_do_not_change_with_live_filter():
    model = FilterWithProperties()
    saver = Saver(model, save_current=True)
    model._state[:] = [3.0, 4.0]
    model._metadata["history"].append(2)
    saver.save()
    model._state[:] = [5.0, 6.0]
    model._metadata["history"].append(3)
    np.testing.assert_array_equal(saver.state, [[1.0, 2.0], [3.0, 4.0]])
    assert saver.metadata == [{"history": [1]}, {"history": [1, 2]}]
    saver.state[0][0] = 99
    np.testing.assert_array_equal(model.state, [5.0, 6.0])
    np.testing.assert_array_equal(saver.state[1], [3.0, 4.0])


def test_ignored_property_is_never_read():
    class Model:
        @property
        def ignored(self):
            raise AssertionError("ignored property accessed")

    saver = Saver(Model(), ignore=("ignored",), save_current=True)
    assert len(saver) == 1
