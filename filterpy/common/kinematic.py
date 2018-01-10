# -*- coding: utf-8 -*-

"""Copyright 2018 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import math
import numpy as np
import scipy as sp
from kalman import KalmanFilter


def kinematic_state_transition(order, dt):
    """ create a state transition matrix of a given order for a given time
    step `dt`.
    """
    assert order >= 0 and int(order) == order, "order must be an int >= 0"

    # hard code common cases for computational efficiency
    if order == 0:
        return np.array([[1.]])
    if order == 1:
        return np.array([[1., dt],
                         [0., 1.]])
    if order == 2:
        return np.array([[1., dt, 0.5*dt*dt],
                         [0., 1., dt],
                         [0., 1., dt]])

    # grind it out computationally....
    N = order + 1

    F = np.zeros((N, N))
    # compute highest order row
    for n in range(N):
        F[0, n] = float(dt**n) / math.factorial(n)

    # copy with a shift to get lower order rows
    for j in range(1, N):
        F[j, j:] = F[0, 0:-j]

    return F


def kinematic_kf(dim, order, dt=1.):
    """ Returns a KalmanFilter using newtonian kinematics for an arbitrary
    number of dimensions and order. So, for example, a constant velocity
    filter in 3D space would be created with

    kinematic_kf(3, 1)


    which will set the state `x` to be interpreted as

    [x, x', y, y', z, z'].T

    As another example, a 2D constant jerk is created with

    kinematic_kf(2, 3)


    Assumes that the measurement z is position in each dimension. If this is not
    true you will have to alter the H matrix by hand.

    P, Q, R are all set to the Identity matrix.

    H is assigned assuming the measurement is position, one per dimension `dim`.

    Parameters
    ----------

    dim : int
        number of dimensions

    order : int, >= 1
        order of the filter. 2 would be a const acceleration model.

    dim_z : int, default 1
        size of z vector *per* dimension `dim`. Normally should be 1

    dt : float, default 1.0
        Time step. Used to create the state transition matrix


    """

    kf = KalmanFilter(dim*order, dim)

    F = kinematic_state_transition(order, dt)
    diag = [F] * dim
    kf.F = sp.linalg.block_diag(*diag)

    kf.H = np.zeros((dim, dim*order))
    for i in range(dim):
        kf.H[i, i * order] = 1.

    return kf

kf = kinematic_kf(3,2)
print(kf.H)




