# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 20:52:38 2014

@author: rlabbe
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from filterpy.common import linear_ode_discretation
from numpy import array

def near_eq(x,y):
    return abs(x-y) < 1.e-18

def test_linear_ode():

    F = array([[0,0,1,0,0,0],
               [0,0,0,1,0,0],
               [0,0,0,0,1,0],
               [0,0,0,0,0,1],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]], dtype=float)

    L = array ([[0,0],
                [0,0],
                [0,0],
                [0,0],
                [1,0],
                [0,1]], dtype=float)

    q = .2
    Q = array([[q, 0],[0, q]])
    dt = 0.5
    A,Q = linear_ode_discretation(F, L, Q, dt)

    val = [1, 0, dt, 0, 0.5*dt**2, 0]

    for i in range(6):
        assert val[i] == A[0,i]

    for i in range(6):
        assert val[i-1] == A[1,i] if i > 0 else A[1,i] == 0

    for i in range(6):
        assert val[i-2] == A[2,i] if i > 1 else A[2,i] == 0

    for i in range(6):
        assert val[i-3] == A[3,i] if i > 2 else A[3,i] == 0


    for i in range(6):
        assert val[i-4] == A[4,i] if i > 3 else A[4,i] == 0

    for i in range(6):
        assert val[i-5] == A[5,i] if i > 4 else A[5,i] == 0

    assert near_eq(Q[0,0], (1./20)*(dt**5)*q)
    assert near_eq(Q[0,1], 0)
    assert near_eq(Q[0,2], (1/8)*(dt**4)*q)
    assert near_eq(Q[0,3], 0)
    assert near_eq(Q[0,4], (1./6)*(dt**3)*q)
    assert near_eq(Q[0,5], 0)

if __name__ == "__main__":
    test_linear_ode()

    F = array([[0,0,1,0,0,0],
               [0,0,0,1,0,0],
               [0,0,0,0,1,0],
               [0,0,0,0,0,1],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]], dtype=float)

    L = array ([[0,0],
                [0,0],
                [0,0],
                [0,0],
                [1,0],
                [0,1]], dtype=float)

    q = .2
    Q = array([[q, 0],[0, q]])
    dt = 1/30
    A,Q = linear_ode_discretation(F, L, Q, dt)

    print(Q)