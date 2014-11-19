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
from numpy.linalg import inv
from numpy import dot, zeros, eye, asarray
from filterpy.common import dot3, dot4


class FixedLagSmoother(object):
    """ Fixed Lag Kalman smoother.

    DO NOT USE: NOT DEBUGGED.

    Computes a smoothed sequence from a set of measurements.
    
    
    References
    ----------
    Simon, Dan. "Optimal State Estimation," John Wiley & Sons pp 274-8 (2006).
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
        self.F = 0                # state transition matrix
        self.H = 0                # Measurement function
        self.R = eye(dim_z)       # state uncertainty
        self.K = 0                # kalman gain
        self.residual = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)



    def smooth(self, Zs, N):
        
        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        P = self.P
        x = self.x
        Q = self.Q
        
        PCol = zeros((self.dim_x, self.dim_x, N+1))
        PColOld = zeros((self.dim_x, self.dim_x, N+1))
        PSmooth = zeros((self.dim_x, self.dim_x, N+1))
        PSmoothOld = zeros((self.dim_x, self.dim_x, N+1))
        
        xSmooth = zeros((len(Zs), self.dim_x, 1))
        
        xhat = []
        for zi, z in enumerate(Zs):
            x = dot(F, x)
        
            inn = z - dot(H, x)
            S = dot(H, dot(P, H.T)) + R
            SI = linalg.inv(S)
            K = dot(F, dot(P, dot(H.T, SI)))

            x = x + dot(K, inn)
            xSmooth[zi,:,:] = x.copy()
    
            
            xhat.append (x.copy())

            PColOld[:,:,0] = P.copy()
            PSmoothOld[:,:,0] = P.copy()
            
            LHS =  dot (F, dot(P, H.T))
            RHS = dot (H, dot (P, F.T))
            P = dot (F, dot(P, F.T)) - dot (LHS, dot (SI, RHS)) + Q
            
            if zi < N:
                continue
            
            for i in range (N):
                L = dot3(PColOld[:,:,i], H.T, SI)
                PSmooth[:,:,i+1] = PSmoothOld[:,:,i] - dot3(PColOld[:,:,i], H.T, L.T).dot(F.T)
                PCol[:,:,i+1] = dot(PColOld[:,:,i], (F - dot(K,H)).T)
                si = zi-i-1
                if si >= 0 and si+1 < len(Zs):
                    print(zi, si, si+1)
                    xSmooth[si,:,:] = xSmooth[si+1,:,:] + dot(L, inn)

            PSmoothOld = PSmooth.copy()
            PColOld = PCol.copy()
            
        return asarray(xhat), xSmooth


    def smooth2(self, Zs, N):
        
        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        P = self.P
        x = self.x
        Q = self.Q
        
        PCol    = zeros((N+1, self.dim_x, self.dim_x))
        #PSmooth = zeros((N+1, self.dim_x, self.dim_x))
        xSmooth = zeros((len(Zs), self.dim_x, 1))
        xSmooth[0] = x.copy()
        
        xhat = []
        for zi, z in enumerate(Zs):
            
            P00 = P
            y =  z - dot(H,x)

            SI = inv(dot3(H, P, H.T) + R)
            
            L = dot4(F, P, H.T, SI)
            P = dot3(F, P, (F-dot(L,H)).T) + Q
            x = dot(F,x) + dot(L, y)
            
            xhat.append (x.copy())
            xSmooth[zi] = x.copy()

            
            #PSmooth[0] = P00.copy()
            PCol[0]    = P00.copy()

            #compute invariants
            HSI = dot(H.T, SI)
            F_LH = (F - dot(L,H)).T
            PS = P.copy()

            
            
            for i in range (N):
                K = dot3(PCol[i], H.T, SI)
                Pi = dot(PS, F_LH)
                
                
                #PSmooth[i+1] = PSmooth[i] - dot4(PCol[i], H.T, L.T, F.T)
                PCol[i+1] = dot(PCol[i], F_LH)
                
                si = zi-i-1
                if si >= 0 and si+1 < len(Zs):
                    xSmooth[si] = xSmooth[si-1] + dot(L, y)

        #for i in range(len(Zs)-2, 0, -1):
        #    xSmooth[i+1] = xSmooth[i]
            
        return asarray(xhat), xSmooth
