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

from numpy import array, asarray, dot, ones, outer, sum, zeros

class MMAEFilterBank(object):

    def __init__(self, filters, p, dim_x, H=None):

        assert len(filters) == len(p)
        assert dim_x > 0

        self.filters = filters
        self.p = asarray(p)
        self.dim_x = dim_x
        self._x = None
        if H is not None:
            self.H = H
        else:
            self.H = ones(len(filters))

    @property
    def x(self):
        """ The estimated state of the bank of filters."""
        return self._x

    @property
    def P(self):
        """ Estimated covariance of the bank of filters."""
        return self._P

    def predict(self, u=0):
        """ Predict next position using the Kalman filter state propagation
        equations for each filter in the bank.

        **Parameters**

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        for f in self.filters:
            f.predict(u)

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        **Parameters**

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array,  or None
            Optionally provide H to override the measurement function for this
            one call, otherwise  self.H will be used.

        """

        for i, f in enumerate(self.filters):
            f.update(z, R, H)
            self.p[i] *= f.likelihood    # prior * likelihood

        self.p /= sum(self.p) # normalize

        # compute estimated state and covariance of the bank of filters.
        self._P = zeros(self.filters[0].P.shape)

        is_row_vector = (self.filters[0].x.ndim == 1)
        if is_row_vector:

            self._x = zeros(self.dim_x)
            for f, p, h in zip(self.filters, self.p, self.H):
                self._x += dot(dot(f.x, h), p)
        else:
            self._x = zeros((self.dim_x, 1))
            for f, p, h in zip(self.filters, self.p, self.H):
                self._x = zeros((self.dim_x, 1))
                self._x += dot(dot(h, f.x), p)


        try:
            for x, f, p in zip(self._x, self.filters, self.p):
                y = f.x - x
                self._P += p*[outer(y, y) + f.P]
        except:
            # I have no idea how to compute P if the dimensions are different
            # shapes!
            self._P = None
            return

