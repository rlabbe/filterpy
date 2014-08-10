# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from filterpy.gh import GHFilter, least_squares_parameters
from numpy import array


def test_least_squares():
    
    """ there is an alternative form for computing h for the least squares.
    It works for all but the very first term (n=0). Use it to partially test
    the output of least_squares_parameters(). This test does not test that
    g is correct"""
    
    for n in range (1, 100):
        g,h = least_squares_parameters(n)
       
        h2 = 4 - 2*g - (4*(g-2)**2 - 3*g**2)**.5
        
        assert abs(h2-h) < 1.e-12

        
def test_1d_array():
    f1 = GHFilter (0, 0, 1, .8, .2)
    f2 = GHFilter (array([0]), array([0]), 1, .8, .2)

    # test both give same answers, and that we can
    # use a scalar for the measurment    
    for i in range(1,10):
        f1.update(i)
        f2.update(i)
     
        assert f1.x == f2.x[0]       
        assert f1.dx == f2.dx[0] 
        
        assert f1.VRF() == f2.VRF()

    # test using an array for the measurement        
    for i in range(1,10):
        f1.update(i)
        f2.update(array([i]))
     
        assert f1.x == f2.x[0]       
        assert f1.dx == f2.dx[0] 
        
        assert f1.VRF() == f2.VRF()
        
    
def test_2d_array():
    """ test using 2 independent variables for the
    state variable.
    """
    
    f = GHFilter(array([0,1]), array([0,0]), 1, .8, .2)
    f0 = GHFilter(0, 0, 1, .8, .2)
    f1 = GHFilter(1, 0, 1, .8, .2)
    
    # test using scalar in update (not normal, but possible)
    for i in range (1,10):
        f.update (i)
        f0.update(i)
        f1.update(i)
        
        assert f.x[0] == f0.x
        assert f.x[1] == f1.x
        
        assert f.dx[0] == f0.dx
        assert f.dx[1] == f1.dx
        
    # test using array for update (typical scenario)
    f = GHFilter(array([0,1]), array([0,0]), 1, .8, .2)
    f0 = GHFilter(0, 0, 1, .8, .2)
    f1 = GHFilter(1, 0, 1, .8, .2)
    
    for i in range (1,10):
        f.update (array([i, i+3]))
        f0.update(i)
        f1.update(i+3)
        
        assert f.x[0] == f0.x
        assert f.x[1] == f1.x
        
        assert f.dx[0] == f0.dx
        assert f.dx[1] == f1.dx
        
        assert f.VRF() == f0.VRF()
        assert f.VRF() == f1.VRF()
    
        

if __name__ == "__main__":
    
    test_least_squares()
    test_1d_array()
    test_2d_array()
    