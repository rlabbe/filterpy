ExtendedKalmanFilter
====================

Introduction and Overview
-------------------------

Implements a extended Kalman filter. For now the best documentation
is my free book Kalman and Bayesian Filters in Python [1]_

The test files in this directory also give you a basic idea of use,
albeit without much description.

In brief, you will first construct this object, specifying the size of the
state vector with `dim_x` and the size of the measurement vector that you
will be using with `dim_z`. These are mostly used to perform size checks
when you assign values to the various matrices. For example, if you
specified `dim_z=2` and then try to assign a 3x3 matrix to R (the
measurement noise matrix you will get an assert exception because R
should be 2x2. (If for whatever reason you need to alter the size of things
midstream just use the underscore version of the matrices to assign
directly: your_filter._R = a_3x3_matrix.)


After construction the filter will have default matrices created for you,
but you must specify the values for each. It's usually easiest to just
overwrite them rather than assign to each element yourself. This will be
clearer in the example below. All are of type numpy.array.


**References**


.. [1] Labbe, Roger. "Kalman and Bayesian Filters in Python".

github repo:
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

read online:
    http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb

PDF version (often lags the two sources above)
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Kalman_and_Bayesian_Filters_in_Python.pdf


-------

.. automodule:: filterpy.kalman

.. autoclass:: ExtendedKalmanFilter
    :members:

    .. automethod:: __init__
