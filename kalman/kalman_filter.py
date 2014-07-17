# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

This is licensed under an MIT license. See the readme.MD file
for more information.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import numpy.random as random

class KalmanFilter:

    def __init__(self, dim_x, dim_z):
        """ Create a Kalman filter with 'dim_x' state variables, and
        'dim_z' measurements. You are responsible for setting the various
        state variables to reasonable values; the defaults below will
        not give you a functional filter.
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

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


    def update(self, Z, R=None):
        """
        Add a new measurement (Z) to the kalman filter. If Z is None, nothing
        is changed.

        Optionally provide R to override the measurement noise for this
        one call, otherwise  self.R will be used.

        self.residual, self.S, and self.K are stored in case you want to
        inspect these variables. Strictly speaking they are not part of the
        output of the Kalman filter, however, it is often useful to know
        what these values are in various scenarios.
        """

        if Z is None:
            return

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


    def batch_filter(self, Zs, Rs=None, update_first=False):
        """ Batch processes a sequences of measurements.

        Parameters
        ----------
        Zs : list-like
            list of measurements at each time step `self.dt` Missing
            measurements must be represented by 'None'.

        Rs : list-like, optional

            optional list of values to use for the measurement error
            covariance; a value of None in any position will cause the filter
            to use `self.R` for that time step.

        update_first : bool, optional,

            controls whether the order of operations is update followed by
            predict, or predict followed by update. Default is predict->update.

        Returns
        -------
        
        means: np.array((n,dim_x,1))
            array of the state for each time step. Each entry is an np.array.
            In other words `means[k,:]` is the state at step `k`.
            
        covariance: np.array((n,dim_x,dim_x))
            array of the covariances for each time step. In other words 
            `covariance[k,:,:]` is the covariance at step `k`.
        """

        n = np.size(Zs,0)
        if Rs is None:
            Rs = [None]*n

        # mean estimates from Kalman Filter
        means = np.zeros((n,self.dim_x,1))

        # state covariances from Kalman Filter
        covariances = np.zeros((n,self.dim_x,self.dim_x))

        if update_first:
            for i,(z,r) in enumerate(zip(Zs,Rs)):
                self.update(z,r)
                means[i,:] = self.x
                covariances[i,:,:] = self.P
                self.predict()
        else:
            for i,(z,r) in enumerate(zip(Zs,Rs)):
                self.predict()
                self.update(z,r)

                means[i,:] = self.x
                covariances[i,:,:] = self.P

        return (means, covariances)


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

    zs = []
    for t in range (100):
        # create measurement = t plus white noise
        z = t + random.randn()*20
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        # save data
        results.append (f.x[0,0])
        measurements.append(z)


    # now do a batch run with the stored z values so we can test that
    # it is working the same as the recursive implementation.
    # give slightly different P so result is slightly different
    f.x = np.array([[2.,0]]).T
    f.P = np.eye(2)*100. 
    m,c = f.batch_filter(zs,update_first=False)

    # plot data
    p1, = plt.plot(measurements,'r', alpha=0.5)
    p2, = plt.plot (results,'b')
    p4, = plt.plot(m[:,0], 'm')
    p3, = plt.plot ([0,100],[0,100], 'g') # perfect result
    plt.legend([p1,p2, p3, p4], 
               ["noisy measurement", "KF output", "ideal", "batch"], 4)


    plt.show()