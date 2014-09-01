# -*- coding: utf-8 -*-

"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from numpy import asarray, isscalar, eye, dot
from functools import reduce


def dot3(A,B,C):
    """ Returns the matrix multiplication of A*B*C"""
    return dot(A, dot(B,C))


def dotn(*args):
    """ returns the matrix multiplication of 2 or more matrices"""
    return reduce(dot, args)
    

def setter(value, dim_x, dim_y):
    """ returns a copy of 'value' as an numpy.array with dtype=float. Throws 
    exception if the array is not dimensioned correctly. Value may be any
    type which converts to numpy.array (list, np.array, np.matrix, etc)
    """
    v = asarray(value, dtype=float)
    if v is value:
        v = value.copy()
    if v.shape != (dim_x, dim_y):
        raise Exception('must have shape ({},{})'.format(dim_x, dim_y))
    return v


def setter_scalar(value, dim_x):
    """ returns a copy of 'value' as an numpy.array with dtype=float. Throws 
    exception if the array is not dimensioned correctly. Value may be any
    type which converts to numpy.array (list, np.array, np.matrix, etc),
    or a scalar, in which case we create a diagonal matrix with each
    diagonal element == value.
    """
    if isscalar(value):
        v = eye(dim_x) * value
    else:
        v = asarray(value, dtype=float)
        
    if v is value:
        v = value.copy()
    if v.shape != (dim_x, dim_x):
        raise Exception('must have shape ({},{})'.format(dim_x, dim_x))
    return v            
