# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 09:22:36 2014

@author: rlabbe
"""

from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

def test_noisy_1d():
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


if __name__ == "__main__":
    test_noisy_1d()