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
import numpy as np
from scipy.linalg import inv
from numpy import dot, zeros, eye, asarray
from filterpy.common import setter, setter_scalar, dot3



class InformationFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        """ Create a Information filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        **Parameters**

        dim_x : int
            Number of state variables for the  filter. For example, if you
            are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        dim_u : int (optional)
            size of the control input, if it is being used.
            Default value of 0 indicates it is not used.
        """

        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self._x = zeros((dim_x,1)) # state
        self._P_inv = eye(dim_x)   # uncertainty covariance
        self._Q = eye(dim_x)       # process uncertainty
        self._B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._F_inv = 0            # state transition matrix
        self._H = 0                # Measurement function
        self._R_inv = eye(dim_z)   # state uncertainty

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self._K = 0 # kalman gain
        self._y = zeros((dim_z, 1))
        self._S = 0 # system uncertainty in measurement space

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)
        self._no_information = False


    def update(self, z, R_inv=None):
        """
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.

        **Parameters**

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        """

        if z is None:
            return

        if R_inv is None:
            R_inv = self._R_inv
        elif np.isscalar(R_inv):
            R_inv = eye(self.dim_z) * R_inv

        # rename for readability and a tiny extra bit of speed
        H = self._H
        H_T = H.T
        P_inv = self._P_inv
        x = self._x

        if self._no_information:
            self._x = dot(P_inv, x) + dot3(H_T, R_inv, z)
            self._P_inv = P_inv + dot3(H_T, R_inv, H)

        else:       # y = z - Hx
            # error (residual) between measurement and prediction
            self._y = z - dot(H, x)

            # S = HPH' + R
            # project system uncertainty into measurement space
            self._S = P_inv + dot(H_T, R_inv).dot (H)
            self._K = dot3(inv(self._S), H_T, R_inv)

            # x = x + Ky
            # predict new x with residual scaled by the kalman gain
            self._x = x + dot(self._K, self._y)
            self._P_inv = P_inv + dot3(H_T, R_inv, H)


    def predict(self, u=0):
        """ Predict next position.

        **Parameters**

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """

        # x = Fx + Bu

        A = dot3(self._F_inv.T, self._P_inv, self._F_inv)
        try:
            AI = inv(A)
            invertable = True
            if self._no_information:
                try:
                    self._x = dot(inv(self._P_inv), self._x)
                except:
                    self._x = dot(0, self._x)
                self._no_information = False
        except:
            invertable = False
            self._no_information  = True

        if invertable:
            self._x = dot(self._F, self.x) + dot(self._B, u)
            self._P_inv = inv(AI + self._Q)
        else:
            I_PF = self._I - dot(self._P_inv,self._F_inv)
            FTI = inv(self._F.T)
            FTIX = dot(FTI, self._x)
            print('Q=', self._Q)
            print('A=', A)
            AQI = inv(A + self._Q)
            self._x = dot(FTI, dot3(I_PF, AQI, FTIX))


    def batch_filter(self, zs, Rs=None, update_first=False):
        """ Batch processes a sequences of measurements.

        **Parameters**

        zs : list-like
            list of measurements at each time step `self.dt` Missing
            measurements must be represented by 'None'.

        Rs : list-like, optional
            optional list of values to use for the measurement error
            covariance; a value of None in any position will cause the filter
            to use `self.R` for that time step.

        update_first : bool, optional,
            controls whether the order of operations is update followed by
            predict, or predict followed by update. Default is predict->update.

        **Returns**

        means: np.array((n,dim_x,1))
            array of the state for each time step. Each entry is an np.array.
            In other words `means[k,:]` is the state at step `k`.

        covariance: np.array((n,dim_x,dim_x))
            array of the covariances for each time step. In other words
            `covariance[k,:,:]` is the covariance at step `k`.
        """

        raise "this is not implemented yet"

        ''' this is a copy of the code from kalman_filter, it has not been
        turned into the informatio filter yet. DO NOT USE.'''

        n = np.size(zs,0)
        if Rs is None:
            Rs = [None]*n

        # mean estimates from Kalman Filter
        means = zeros((n,self.dim_x,1))

        # state covariances from Kalman Filter
        covariances = zeros((n,self.dim_x,self.dim_x))

        if update_first:
            for i,(z,r) in enumerate(zip(zs,Rs)):
                self.update(z,r)
                means[i,:] = self._x
                covariances[i,:,:] = self._P
                self.predict()
        else:
            for i,(z,r) in enumerate(zip(zs,Rs)):
                self.predict()
                self.update(z,r)

                means[i,:] = self._x
                covariances[i,:,:] = self._P

        return (means, covariances)


    def get_prediction(self, u=0):
        """ Predicts the next state of the filter and returns it. Does not
        alter the state of the filter.

        **Parameters**

        u : np.array
            optional control input

        **Returns**

        (x, P)
            State vector and covariance array of the prediction.
        """
        raise "Not implemented yet"

        x = dot(self._F, self._x) + dot(self._B, u)
        P = dot3(self._F, self._P, self._F.T) + self.Q
        return (x, P)


    def residual_of(self, z):
        """ returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        raise "Not implemented yet"
        return z - dot(self._H, self._x)


    def measurement_of_state(self, x):
        """ Helper function that converts a state into a measurement.

        **Parameters**

        x : np.array
            kalman state vector

        **Returns**

        z : np.array
            measurement corresponding to the given state
        """
        raise "Not implemented yet"
        return dot(self._H, x)


    @property
    def Q(self):
        """ Process uncertainty"""
        return self._Q


    @Q.setter
    def Q(self, value):
        self._Q = setter_scalar(value, self.dim_x)

    @property
    def P_inv(self):
        """ inverse covariance matrix"""
        return self._P_inv


    @P_inv.setter
    def P_inv(self, value):
        self._P_inv = setter_scalar(value, self.dim_x)


    @property
    def R_inv(self):
        """ measurement uncertainty"""
        return self._R_inv


    @R_inv.setter
    def R_inv(self, value):
        self._R_inv = setter_scalar(value, self.dim_z)

    @property
    def H(self):
        return self._H


    @H.setter
    def H(self, value):
        self._H = setter(value, self.dim_z, self.dim_x)


    @property
    def F(self):
        return self._F


    @F.setter
    def F(self, value):
        self._F = setter(value, self.dim_x, self.dim_x)
        self._F_inv = inv(self._F)

    @property
    def B(self):
        """ control transition matrix"""
        return self._B


    @B.setter
    def B(self, value):
        """ control transition matrix"""
        self._B = setter (value, self.dim_x, self.dim_u)


    @property
    def x(self):
        return self._x


    @x.setter
    def x(self, value):
        self._x = setter(value, self.dim_x, 1)

    @property
    def K(self):
        """ Kalman gain """
        return self._K

    @property
    def y(self):
        """ measurement residual (innovation) """
        return self._y

    @property
    def S(self):
        """ system uncertainy in measurement space """
        return self._S



