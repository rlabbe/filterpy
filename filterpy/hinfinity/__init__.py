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

#pylint: disable=wildcard-import

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ["hinfinity_filter"]

from .hinfinity_filter import *
