from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:27:25 2015

@author: rlabbe
"""

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


#***********************************************************
# WARNING!!!!
# I am still working on this; it is not an official part of
# the package yet
#***********************************************************

import numpy as np
from scipy.linalg import inv
from numpy import dot, zeros, eye
from filterpy.common import dot3, dot4, dotn


class FixedPointSmoother(object):
    """ Fixed Point Kalman smoother.
    **Methods**
    """


    def __init__(self, dim_x, dim_z, j, N=None):
        """ Create a fixed point Kalman filter smoother. You are responsible
        for setting the various state variables to reasonable values;
        the defaults below will not give you a functional filter.

        **Parameters**

        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        j: int
            step used for the fixed point

        N : int, optional
            If provided, the size of the lag. Not needed if you are only
            using smooth_batch() function. Required if calling smooth()
        """

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N     = N
        self.j     = j

        self.x = zeros((dim_x,1)) # state
        self.x_s = zeros((dim_x,1)) # smoothed state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.F = 0                # state transition matrix
        self.H = 0                # Measurement function
        self.R = eye(dim_z)       # state uncertainty
        self.K = 0                # kalman gain
        self.residual = zeros((dim_z, 1))

        self.B = 0

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self.count = 0

        if N is not None:
            self.xSmooth = []


    def smooth(self, z, u=0):
        """ Predict next position using the Kalman filter state propagation
        equations.

        **Parameters**

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None in
            any position will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None in
            any position will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None in
            any position will cause the filter to use `self.Q`.
        """

        B = self.B
        F = self.F
        Q = self.Q
        R = self.R
        H = self.H

        if self.count < self.j:
            self.x = dot(F, self.x) + dot(B, u)
            self.P = dot3(F, self.P, F.T) + Q
            self.y = z - dot(H, self.x)
            S = dot3(H, self.P, H.T) + R
            K = dot3(self.P, H.T, inv(S))
            self.x += dot(K, self.y)
            I_KH = self._I - dot(K, H)
            self.P = dot3(I_KH, self.P, I_KH.T) + dot3(K, R, K.T)
        else:
            if self.count == self.j:
                self.Pjk = self.P.copy() # cross covariance between j and k
                self.Pj = self.P.copy()
                self.xj = self.x.copy()


            self.S = dot3(H, self.P, H.T) + R
            SI = inv(self.S)
            self.K = dot4(F, self.P, H.T, SI)
            self.Kj = dot3(self.Pjk, H.T, SI)

            self.y = z - dot(H, self.x)
            self.x += dot(self.K, self.y)
            self.xj += dot(self.Kj, self.y)

            F_KH_T = (F - dot(self.K, H)).T
            self.P = dot3(F, self.P, F_KH_T) + Q
            self.Pj += dot3(self.Pjk, H.T, self.Kj.T)
            self.Pjk = dot(self.Pjk, F_KH_T)

        self.count += 1



class GrewalFixedPointSmoother(object):
    """ Fixed Point Kalman smoother.
    **Methods**
    """


    def __init__(self, dim_x, dim_z, j, N=None):
        """ Create a fixed point Kalman filter smoother. You are responsible
        for setting the various state variables to reasonable values;
        the defaults below will not give you a functional filter.

        **Parameters**

        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        j: int
            step used for the fixed point

        N : int, optional
            If provided, the size of the lag. Not needed if you are only
            using smooth_batch() function. Required if calling smooth()
        """

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N     = N
        self.j     = j

        self.x = zeros((dim_x,1)) # state
        self.x_s = zeros((dim_x,1)) # smoothed state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.F = 0                # state transition matrix
        self.H = 0                # Measurement function
        self.R = eye(dim_z)       # state uncertainty
        self.K = 0                # kalman gain
        self.residual = zeros((dim_z, 1))

        self.B = 0

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self.count = 0

        if N is not None:
            self.xSmooth = []


    def smooth(self, z, u=0):
        """ Predict next position using the Kalman filter state propagation
        equations.

        **Parameters**

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None in
            any position will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None in
            any position will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None in
            any position will cause the filter to use `self.Q`.
        """

        B = self.B
        F = self.F
        Q = self.Q
        R = self.R
        H = self.H

        if self.count < self.j:
            self.x = dot(F, self.x) + dot(B, u)
            self.P = dot3(F, self.P, F.T) + Q
            self.y = z - dot(H, self.x)
            S = dot3(H, self.P, H.T) + R
            K = dot3(self.P, H.T, inv(S))
            self.x += dot(K, self.y)
            I_KH = self._I - dot(K, H)
            self.P = dot3(I_KH, self.P, I_KH.T) + dot3(K, R, K.T)
        else:
            if self.count == self.j:
                self.Pjk = self.P.copy() # cross covariance between j and k
                self.Pj = self.P.copy()
                self.xj = self.x.copy()


            self.S = dot3(H, self.P, H.T) + R
            SI = inv(self.S)
            self.K = dot4(F, self.P, H.T, SI)
            self.Kj = dot3(self.Pjk, H.T, SI)

            self.y = z - dot(H, self.x)
            self.x += dot(self.K, self.y)
            self.xj += dot(self.Kj, self.y)

            F_KH_T = (F - dot(self.K, H)).T
            self.P = dot3(F, self.P, F_KH_T) + Q
            self.Pj += dot3(self.Pjk, H.T, self.Kj.T)
            self.Pjk = dot(self.Pjk, F_KH_T)

        self.count += 1


