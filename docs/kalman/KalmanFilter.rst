KalmanFilter
============

Implements a linear Kalman filter. For now the best documentation
is my free book Kalman and Bayesian Filters in Python [2]_

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

These are the matrices (instance variables) which you must specify.
All are of type numpy.array (do NOT use numpy.matrix) If dimensional
analysis allows you to get away with a 1x1 matrix you may also use a
scalar. All elements must have a type of float.


**Instance Variables**

You will have to assign reasonable values to all of these before
running the filter. All must have dtype of float.

x : ndarray (dim_x, 1), default = [0,0,0...0]
    filter state estimate

P : ndarray (dim_x, dim_x), default eye(dim_x)
    covariance matrix

Q : ndarray (dim_x, dim_x), default eye(dim_x)
    Process uncertainty/noise

R : ndarray (dim_z, dim_z), default eye(dim_z)
    measurement uncertainty/noise

H : ndarray (dim_z, dim_x)
    measurement function

F : ndarray (dim_x, dim_x)
    state transition matrix

B : ndarray (dim_x, dim_u), default 0
    control transition matrix


**Optional Instance Variables**

alpha : float

Assign a value > 1.0 to turn this into a fading memory filter.


**Read-only Instance Variables**


K : ndarray
    Kalman gain that was used in the most recent update() call.

y : ndarray
    Residual calculated in the most recent update() call. I.e., the
    different between the measurement and the current estimated state
    projected into measurement space (z - Hx)

S : ndarray
    System uncertainty projected into measurement space. I.e., HPH' + R.
    Probably not very useful, but it is here if you want it.

likelihood : float
    Likelihood of last measurment update.

log_likelihood : float
    Log likelihood of last measurment update.


**Example**


Here is a filter that tracks position and velocity using a sensor that only
reads position.

First construct the object with the required dimensionality.

.. code::

    from filterpy.kalman import KalmanFilter
    f = KalmanFilter (dim_x=2, dim_z=1)


Assign the initial value for the state (position and velocity). You can do this
with a two dimensional array like so:

.. code::

    f.x = np.array([[2.],    # position
                    [0.]])   # velocity

or just use a one dimensional array, which I prefer doing.

.. code::

    f.x = np.array([2., 0.])


Define the state transition matrix:

.. code::

    f.F = np.array([[1.,1.],
                    [0.,1.]])

Define the measurement function:

.. code::

    f.H = np.array([[1.,0.]])

Define the covariance matrix. Here I take advantage of the fact that
P already contains np.eye(dim_x), and just multiply by the uncertainty:

.. code::

    f.P *= 1000.

I could have written:

.. code::

    f.P = np.array([[1000.,    0.],
                    [   0., 1000.] ])

You decide which is more readable and understandable.

Now assign the measurement noise. Here the dimension is 1x1, so I can
use a scalar

.. code::

    f.R = 5

I could have done this instead:

.. code::

    f.R = np.array([[5.]])

Note that this must be a 2 dimensional array, as must all the matrices.

Finally, I will assign the process noise. Here I will take advantage of
another FilterPy library function:

.. code::

    from filterpy.common import Q_discrete_white_noise
    f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)


Now just perform the standard predict/update loop:

while some_condition_is_true:

.. code::

    z = get_sensor_reading()
    f.predict()
    f.update(z)

    do_something_with_estimate (f.x)


**Procedural Form**

This module also contains stand alone functions to perform Kalman filtering.
Use these if you are not a fan of objects.

**Example**

.. code::

    while True:
        z, R = read_sensor()
        x, P = predict(x, P, F, Q)
        x, P = update(x, P, z, R, H)    
    
**References**


.. [2] Labbe, Roger. "Kalman and Bayesian Filters in Python".

github repo:
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

read online:
    http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb

PDF version (often lags the two sources above)
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Kalman_and_Bayesian_Filters_in_Python.pdf



-------

.. automodule:: filterpy.kalman

Kalman filter

.. autoclass:: KalmanFilter
    :members:

    .. automethod:: __init__


.. autofunction:: update
.. autofunction:: predict
.. autofunction:: batch_filter


