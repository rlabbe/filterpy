# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy.linalg import inv
from numpy import dot, zeros, isscalar, outer
from filterpy.common import dot3


"""THIS FILE'S FUNCTIONALIY HAS BEEN MOVED.

RTS smoothing is now a method in the KF classes: KalmanFilter and
UnscentedKalmanFilter. We do not yet have one implemented for the EKF."""

