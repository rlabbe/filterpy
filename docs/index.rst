.. FilterPy documentation master file


Welcome to FilterPy's documentation!
====================================

Contents:

.. toctree::
   :maxdepth: 2


FilterPy is a Python library that implements a number of Bayesian filters,
most notably Kalman filters. I am writing it in conjunction with my book
*Kalman and Bayesian Filters in Python* [1]_, a free book written using
Ipython Notebook, hosted on github, and readable via nbviewer.

As such this library has a strong pedalogical flavor. It is rare that I
choose the most efficient way to calculate something unless it does not
obscure exposition of the concepts of the filtering being done. I will always
opt for clarity over speed. I do not mean to imply that this is a toy; I use
it all of the time in my job.

I mainly develop in Python 3.x, but this should support both Python 2.x and
3.x flavors. At the moment I can not tell you the lowest required version;
I tend to develop on the bleeding edge of the Python releases. I am happy to
receive bug reports if it does not work with older versions, but testing
backwards compatibility is not a high priority at the moment. As the package
matures I will shift my focus in that direction.

FilterPy requires Numpy [2]_ and SciPy [3]_ to work. The tests and examples
also use matplotlib [4]_. For testing I use py.test [5]_.


Installation
------------

FilterPy is available on github (https://github.com/rlabbe/filterpy). However,
it is also hosted on PyPi, and unless you want to be on the bleeding edge of
development I recommend you get it from there. To install from the command line,
merely type:

    $ pip install filterpy

To test the installation, from a python REPL type:

    >>> import filterpy
    >>> filterpy.__version__

and it should display the version number that you installed.



Use
---

There are several submodules, each listed below. But in general you will
need to import which classes and/or functions you need from the correct
submodule, construct the objects, and then execute your code. Something lke

    >>> from filterpy.kalman import KalmanFilter
    >>> kf = KalmanFilter(dim_x=3, dim_z=1)
    


filterpy.kalman Module
++++++++++++++++++++++

These are the classes that are implemented in the filterpy.kalman submodule.
To import one you would use

   >>> from filterpy.kalman import KalmanFilter**
   
   
.. toctree::
   :maxdepth: 1

   KalmanFilter
   ExtendedKalmanFilter
   kalman/FixedLagSmoother
   RTS_smoother
   SquareRootFilter
   InformationFilter
   UnscentedKalmanFilter


filterpy.common Module
++++++++++++++++++++++

   Common


filterpy.gh Module
++++++++++++++++++


filterpy.memory Module
++++++++++++++++++++++


filterpy.hinfinity Module
+++++++++++++++++++++++++


filterpy.leastsq Module
+++++++++++++++++++++++



**References**


.. [1] Labbe, Roger. "Kalman and Bayesian Filters in Python".

github repo:
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

read online:
    http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb

PDF version (often lags the two sources above)
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Kalman_and_Bayesian_Filters_in_Python.pdf

.. [2] NumPy
    http://www.numpy.org

.. [3] SciPy
    http://www.scipy.org

.. [4] matplotlib
    http://http://matplotlib.org/

.. [5] pytest http://pytest.org/latest/




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
