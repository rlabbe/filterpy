# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

Rauch-Tung-Striebal Kalman smoother from the filterpy library.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import numpy.linalg as linalg
from numpy import dot, zeros, eye

class FixedLagSmoother(object):
    """ Fixed Lag Kalman smoother.

    DO NOT USE: NOT DEBUGGED.

    Computes a smoothed sequence from a set of measurements.
    """

    def __init__(self, dim_x, dim_z):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        Parameters
        ----------
        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.
        """

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.u = 0                # control input vector
        self.B = 0
        self.F = 0                # state transition matrix
        self.H = 0                # Measurement function
        self.R = eye(dim_z)       # state uncertainty
        self.K = 0                # kalman gain
        self.residual = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)



    def smooth(self, Ms, N):
        
        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        B = self.B
        u = self.u
        P = self.P
        x = self.x
        Q = self.Q
        
        PCol = zeros((self.dim_x, self.dim_x, N+2))
        PColOld = zeros((self.dim_x, self.dim_x, N+2))
        PSmooth = zeros((self.dim_x, self.dim_x, N+2))
        PSmoothOld = zeros((self.dim_x, self.dim_x, N+2))
        
        xhat = []
        for z in Ms:
            x = dot(F, x) + dot(B, u)
        
            inn = z - dot(H, x)
            S = dot(H, dot(P, H.T)) + R
            SI = linalg.inv(S)
            K = dot(F, dot(P, dot(H.T, SI)))
            KSmooth = K.copy()
            x = x + dot(K, inn)
            #xSmooth = x.copy()
            
            xhat.append (x.copy())

            PColOld[:,:,0] = P.copy()
            PSmoothOld[:,:,0] = P.copy()
            
            LHS =  dot (F, dot(P, H.T))
            RHS = dot (H, dot (P, F.T))
            P = dot (F, dot(P, F.T)) - dot (LHS, dot (SI, RHS)) + Q
            
            for i in range (N+1):
                KSmooth = dot(PColOld[:,:,i], dot(H.T, SI))
                PSmooth[:,:,i+1] = PSmoothOld[:,:,i] - dot(PColOld[:,:,i], dot(H.T, dot(KSmooth.T, H.T)))
                PCol[:,:,i+1] = dot(PColOld[:,:,i], (F - dot(K,H)).T)
                #xSmooth = xSmooth + dot(KSmooth, inn)
                
            PSmoothOld = PSmooth.copy()
            PColOld = PCol.copy()
            
        return xhat
