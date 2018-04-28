
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

#pylint: disable=invalid-name
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
from numpy.random import randn



def get_radar(dt):
    """ Simulate radar range to object at 1K altidue and moving at 100m/s.
    Adds about 5% measurement noise. Returns slant range to the object.
    Call once for each new measurement at dt time from last call.
    """

    if not hasattr(get_radar, "posp"):
        get_radar.posp = 0

    vel = 100  + .5 * randn()
    alt = 1000 + 10 * randn()
    pos = get_radar.posp + vel*dt

    v = 0 + pos* 0.05*randn()
    slant_range = math.sqrt(pos**2 + alt**2) + v
    get_radar.posp = pos

    return slant_range


if __name__ == "__main__":
    for i in range(100):
        print(get_radar(0.1))
