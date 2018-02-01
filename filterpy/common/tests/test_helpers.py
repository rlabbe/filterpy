# -*- coding: utf-8 -*-
"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from filterpy.common import kinematic_kf

import numpy as np
from numpy.linalg import inv




def test_kinematic_filter():
    global kf

    # make sure the default matrices are shaped correctly
    for dim_x in range(1,4):
        for order in range (0, 3):
            kf = kinematic_kf(dim=dim_x, order=order)
            kf.predict()
            kf.update(np.zeros((dim_x, 1)))


    # H is tricky, make sure it is shaped and assigned correctly
    kf = kinematic_kf(dim=2, order=2)
    assert kf.x.shape == (6, 1)
    assert kf.F.shape == (6, 6)
    assert kf.P.shape == (6, 6)
    assert kf.Q.shape == (6, 6)
    assert kf.R.shape == (2, 2)
    assert kf.H.shape == (2, 6)

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]], dtype=float)
    assert np.array_equal(H, kf.H)

    kf = kinematic_kf(dim=3, order=2, order_by_dim=False)
    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=float)
    assert np.array_equal(H, kf.H)


if __name__ == "__main__":

    ITERS = 1000000
    #test_mahalanobis()

    test_kinematic_filter()
