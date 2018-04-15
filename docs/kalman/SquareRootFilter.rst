SquareRootKalmanFilter
======================

Introduction and Overview
-------------------------

This implements a square root Kalman filter. No real attempt has been made
to make this fast; it is a pedalogical exercise. The idea is that by
computing and storing the square root of the covariance matrix we get about
double the significant number of bits. Some authors consider this somewhat
unnecessary with modern hardware. Of course, with microcontrollers being all
the rage these days, that calculus has changed. But, will you really run
a Kalman filter in Python on a tiny chip? No. So, this is for learning.


-------

.. automodule:: filterpy.kalman

Square Root Kalman Filter

.. autoclass:: SquareRootKalmanFilter
    :members:

    .. automethod:: __init__
