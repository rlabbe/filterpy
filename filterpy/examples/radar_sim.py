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

#pylint: skip-file


from numpy.random import randn


class RadarSim(object):
    """
    Simulates the radar signal returns from an object flying
    at a constant altityude and velocity in 1D. Assumes random
    process noise - altitude and velocity varies a bit for each call.
    """


    def __init__(self, dt, pos=0., vel=100., alt=1000.):
        self.dt = dt
        self.pos = pos
        self.vel = vel
        self.alt = alt


    def get_range(self, process_err_pct=0.05):
        """
        Returns slant range to the object. Call once for each
        new measurement at dt time from last call.
        """

        vel = self.vel + 5 * randn()
        alt = self.alt + 10 * randn()

        self.pos += vel*self.dt

        err = (self.pos * process_err_pct) * randn()
        slant_range = (self.pos**2 + alt**2)**.5 + err

        return slant_range
