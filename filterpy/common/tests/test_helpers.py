# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import numpy as np
from filterpy.common import is_arraylike, check_is_array

def test_is_arraylike():
    assert is_arraylike("w34234") == False
    assert is_arraylike("3") == False
    assert is_arraylike(3) == False
    assert is_arraylike(3+1j) == False
    assert is_arraylike([3]) == True
    assert is_arraylike([3, 3]) == True
    assert is_arraylike([3.]) == True
    assert is_arraylike((3,)) == True
    assert is_arraylike(np.asarray([3.])) == True
    assert is_arraylike(np.mat('3')) == True
    assert is_arraylike(np.asarray([[3.]])) == True
    assert is_arraylike(np.asarray([[3.+2j]])) == True
    assert is_arraylike(np.asarray(['3'])) == True

    d = dict()
    assert is_arraylike(d) == False

    s = set()
    assert is_arraylike(s) == False


def test_check_is_array():

    try:
        check_is_array(3, 1, 'z')
        assert False
    except:
        pass

    check_is_array([[[3]]],1,'z')


if __name__ == '__main__':
    test_check_is_array()
    test_is_arraylike()