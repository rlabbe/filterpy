# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 08:03:07 2016

@author: rlabbe
"""


from filterpy.kalman import CubatureKalmanFilter as CKF
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn

'''

def predict(f, x, P, Q):
    n, _ = P.shape
    m = 2 * n
    global S
    S = spherical_radial_sigmas(x, P)

    # evaluate cubature points
    X_ = np.empty((2*n, n))
    for k in range(m):
        X_[k] = f(S[k])


    # predicted state
    x = sum(X_,0)[:,None] / m
    P = np.zeros((n, n))
    xf = x.flatten()
    for k in range(m):
        P += np.outer(X_[k], X_[k]) - np.outer(xf, xf)

    P *= 1 / m
    P += Q

    return x, P, X_


def update(h, z, x, P, R):
    n, _ = P.shape
    nz, _ = z.shape
    m = 2 * n
    #_, S = spherical_radial(h, x, P)

    # evaluate cubature points
    Z_ = np.empty((m, nz))
    for k in range(m):
        Z_[k] = h(S[k])

    # estimate predicted measurement
    z_ = sum(Z_,0)[:,None] / m

    # innovation covariance
    Pz = np.zeros((nz, nz))
    zf = z_.flatten()
    for k in range(m):
        #print(k, np.outer(Z_[k], Z_[k]), np.outer(zf, zf))
        Pz += np.outer(Z_[k], Z_[k]) - np.outer(zf, zf)

    Pz /= m
    Pz += R
    print('Pz', Pz)

    # compute cross variance of the state and the measurements
    Pxz = zeros((n, nz))
    for i in range(m):
        dx = S[i] - x.flatten()
        dz =  Z_[i] - z_
        Pxz += outer(dx, dz)

    Pxz /= m



    K = dot(Pxz, inv(Pz))
    print('K', K.T)

    y = z - z_
    print('y', y)

    x += K @ (z-z_)
    P -= dot3(K, Pz, K.T)
    print('x', x.T)

    return x, P
'''


def test_1d():

    def fx(x, dt):
        F = np.array([[1., dt],
                      [0,  1]])

        return np.dot(F, x)

    def hx(x):
        return np.array([[x[0]]])



    ckf = CKF(dim_x=2, dim_z=1, dt=0.1, hx=hx, fx=fx)

    ckf.x = np.array([[1.], [2.]])
    ckf.P = np.array([[1, 1.1],
                      [1.1, 3]])

    ckf.R = np.eye(1) * .05
    ckf.Q = np.array([[0., 0], [0., .001]])

    dt = 0.1
    points = MerweScaledSigmaPoints(2, .1, 2., -1)
    kf = UKF(dim_x=2, dim_z=1, dt=dt, fx=fx, hx=hx, points=points)


    kf.x = np.array([1, 2])
    kf.P = np.array([[1, 1.1],
                     [1.1, 3]])
    kf.R *= 0.05
    kf.Q = np.array([[0., 0], [0., .001]])


    for i in range(50):
        z = np.array([[i+randn()*0.1]])
        #xx, pp, Sx = predict(f, x, P, Q)
        #x, P = update(h, z, xx, pp, R)
        ckf.predict()
        ckf.update(z)
        kf.predict()
        kf.update(z[0])
        assert abs(ckf.x[0] -kf.x[0]) < 1e-10
        assert abs(ckf.x[1] -kf.x[1]) < 1e-10


if __name__ == "__main__":
    test_1d()