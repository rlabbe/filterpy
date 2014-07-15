# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:02:07 2013

@author: rlabbe
"""

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import numpy.random as random

class KalmanFilter:

    def __init__(self, dim_x, dim_z, use_short_form=False):
        """ Create a Kalman filter with 'dim_x' state variables, and
        'dim_z' measurements. You are responsible for setting the various
        state variables to reasonable values; the defaults below will
        not give you a functional filter.
        """

        self.x = 0 # state
        self.P = np.eye(dim_x) # uncertainty covariance
        self.Q = np.eye(dim_x) # process uncertainty
        self.u = np.zeros((dim_x,1)) # motion vector
        self.B = 0
        self.F = 0 # state transition matrix
        self.H = 0 # Measurement function (maps state to measurements)
        self.R = np.eye(dim_z) # state uncertainty
        
        # identity matrix. Do not alter this. 
        self._I = np.eye(dim_x)
        
        if use_short_form:
            self.update = self.update_short_form


    def update(self, Z, R=None):
        """
        Add a new measurement (Z) to the kalman filter. 
        
        Optionally provide R to override the measurement noise for this 
        one call, otherwise  self.R will be used.
        
        self.residual, self.S, and self.K are stored in case you want to
        inspect these variables. Strictly speaking they are not part of the
        output of the Kalman filter, however, it is often useful to know
        what these values are in various scenarios.
        """
        
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        # error (residual) between measurement and prediction
        self.residual = Z - self.H.dot(self.x)
        
        # project system uncertainty into measurement space 
        self.S = self.H.dot(self.P).dot(self.H.T) + R   

        # map system uncertainty into kalman gain
        self.K = self.P.dot(self.H.T).dot(linalg.inv(self.S)) 

        # predict new x with residual scaled by the kalman gain
        self.x = self.x + self.K.dot(self.residual)                

        KH = self.K.dot(self.H)
        I_KH = self._I - KH
        self.P = (I_KH.dot(self.P.dot(I_KH.T)) + 
                 self.K.dot(self.R.dot(self.K.T)))


    def predict(self):
        """ predict next position """
    
        self.x = self.F.dot(self.x)
        if self.B != 0:
           self.x += self.B.dot(self.u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q


if __name__ == "__main__":
    f = KalmanFilter (dim_x=2, dim_z=1)

    f.x = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    f.F = np.array([[1.,1.],
                    [0.,1.]])    # state transition matrix

    f.H = np.array([[1.,0.]])    # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R *= 5                       # state uncertainty
    f.Q *= 0.0001                 # process uncertainty

    measurements = []
    results = []

    for t in range (100):
        # create measurement = t plus white noise
        z = t + random.randn()*20

        # perform kalman filtering
        f.update(z)
        f.predict()

        # save data
        results.append (f.x[0,0])
        measurements.append(z)

    # plot data
    p1, = plt.plot(measurements,'r')
    p2, = plt.plot (results,'b')
    p3, = plt.plot ([0,100],[0,100], 'g') # perfect result
    plt.legend([p1,p2, p3], ["noisy measurement", "KF output", "ideal"], 4)


    plt.show()