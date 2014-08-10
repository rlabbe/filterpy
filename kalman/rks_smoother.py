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

class RKSSmoother(object):
    """ Rauch-Tung-Striebal Kalman smoother.

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
                
            
            
            

                        
            

        """ Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by a Kalman filter. The usual input
        would come from the output of `KalmanFilter.batch_filter()`.

        Parameters
        ----------

        Ms : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        F : numpy.array
            State transition function of the Kalman filter

        Q : numpy.array
            Process noise of the Kalman filter


        Returns
        -------
        'M' : numpy.array
           smoothed means

        'P' : numpy.array
           smoothed state covariances

        'D' : numpy.array

        """
        M = np.copy(Ms)
        P = np.copy(Ps)
        assert len(M) == len(P)

        n     = np.size(M,0)  # number of measurements
        dim_x = np.size(M,1)  # number of state variables

        D = np.zeros((n,dim_x,dim_x))


        for k in range(n-2,-1,-1):
            P_pred = F.dot(P[k]).dot(F.T) + Q
            #D[k,:,:] = linalg.solve(P[k].dot(F.T).T, P_pred.T)
            D[k,:,:] = P[k].dot(linalg.solve((F.T).T, P_pred.T))
            M[k] = M[k] + D[k].dot(M[k+1] - F.dot(M[k]))
            P[k,:,:] = P[k,:,:] + D[k].dot(P[k+1,:,:] - P_pred).dot(D[k].T)


        return (M,P,D)

