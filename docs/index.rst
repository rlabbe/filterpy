.. FilterPy documentation master file


FilterPy
********

.. toctree::
    :maxdepth: 2


FilterPy is a Python library that implements a number of Bayesian filters,
most notably Kalman filters. I am writing it in conjunction with my book
*Kalman and Bayesian Filters in Python* [1]_, a free book written using
Ipython Notebook, hosted on github, and readable via nbviewer. However, 
it implements a wide variety of functionality that is not described in 
the book.

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
============

Installation with pip (recommended)
-----------------------------------

FilterPy is available on github (https://github.com/rlabbe/filterpy). However,
it is also hosted on PyPi, and unless you want to be on the bleeding edge of
development I recommend you get it from there. To install from the command line,
merely type:

.. code-block:: bash

    $ pip install filterpy

To test the installation, from a python REPL type:

    >>> import filterpy
    >>> filterpy.__version__

and it should display the version number that you installed.


Installation with GitHub
------------------------

You can get the very latest code by getting it from GitHub and then performing
the installation. I will say I am not following particularly stringent version
control discipline. I mostly stay on **master** and commit things that are not
entirely ready for prime-time, mostly because I'm the only one developing.
I do not promise that any check in that is not tagged with a version number is
usable.

.. code-block:: bash

     $ git clone --depth=1 https://github.com/rlabbe/filterpy.git
     $ cd filterpy
     $ python setup.py install

`--depth=1` just gets you the last few revisions that I made, which 
keeps the repo small. If you want the entire repo leave out the `depth`
parameter, or fork the repo if you plan to modify it. 

Use
===

There are several submodules, each listed below. But in general you will
need to import which classes and/or functions you need from the correct
submodule, construct the objects, and then execute your code. Something lke

    >>> from filterpy.kalman import KalmanFilter
    >>> kf = KalmanFilter(dim_x=3, dim_z=1)

I try to provide examples in the help for each class, but this documentation
needs a lot of work. For now I refer you to my book mentioned above if the
documentation is not adequate. Better yet, write an issue on the GitHub
issue tracker. I will respond with an answer as soon as I am online and
available (minutes to a day, normally), and then revise the documentation.
I shouldn't have to be prodded like this, but life is limited. So prod.

Raise issues here: https://github.com/rlabbe/filterpy/issues


FilterPy's Naming Conventions
==============================

A word on variable names. I am an advocate for descriptive variable names. In the Kalman filter literature the measurement noise covariance matrix is called `R`. The name `R` is not descriptive. I could reasonably call it `measurement_noise_covariance`, and I've seen libraries do that. I've chosen not to.

In the end, Kalman filtering is math. To write a Kalman filter you are going to start by sitting down with a piece of paper and doing math. You will be writing and solving normal algebraic equations. Every Kalman filter text and source on the web uses the same  equations. You cannot read about the Kalman filter without seeing this equation

.. math::

    \dot{\mathbf{x}} = \mathbf{Fx} + \mathbf{Gu} + w

One of my goals is to bring you to the point where you can read the original literature on Kalman filtering. For nontrivial problems the difficulty is not the implementation of the equations, but learning how to set up the equations so they solve your problem. In other words, every Kalman filter implements :math:`\dot{\mathbf{x}} = \mathbf{Fx} + \mathbf{Gu} + w`; the difficult part is figuring out what to put in the matrices :math:`\mathbf{F}` and :math:`\mathbf{G}` to make your filter work for your problem. Vast amounts of work have been done to apply Kalman filters in various domains, and it would be tragic to be unable to avail yourself of this research. 

So, like it or not you will need to learn that :math:`\mathbf{F}` is the *state transition matrix* and that :math:`\mathbf{R}` is the *measurement noise covariance*. Once you know that the code will become readable, and until then Kalman filter
math, and all publications and web articles on Kalman filters will be inaccessible to you. 

Finally, I think that mathematical programming is somewhat different than regular programming; what is readable in one domain is not readable in another. `q = x + m` is opaque in a normal context. On the other hand, `x = (.5*a)*t**2 + v_0*t + x_0` is to me the most readable way to program the Newtonian distance equation:

.. math::
     x = \frac{1}{2}at^2 + v_0 t + x_0

We could write it as

.. code-block:: Python

    distance = (.5 * constant_acceleration) * time_delta**2 + 
               initial_velocity * time_delta + initial_distance
    
but I feel that obscures readability. This is debatable for this one equation; but most mathematical programs, and certainly Kalman filters, use systems of equations. I can most easily follow the code, and ensure that it does not have bugs, when it reads as close to the math as possible. Consider this equation from the Kalman filter:

.. math::
    \mathbf{K} = \mathbf{PH}^\mathsf{T}[\mathbf{HPH}^\mathsf{T} + \mathbf{R}]^{-1}

Python code for this would be

.. code-block:: Python

    K = dot(P, H.T).dot(inv(dot(H, P).dot(H.T) + R))
    
It's already a bit hard to read because of the `dot` function calls (required because Python does not yet support an operator for matrix multiplication). But compare this to::

    kalman_gain = (
        dot(apriori_state_covariance, measurement_function_transpose).dot(
        inv(dot(measurement_function, apriori_state_covariance).dot(
        measurement_function_transpose) + measurement_noise_covariance)))

which I adapted from a popular library. I grant you this version has more context, but I cannot glance at this and see what math it is implementing. In particular, the linear algebra :math:`\mathbf{HPH}^\mathsf{T}` is doing something very specific - multiplying :math:`\mathbf{P}` by :math:`\mathbf{H}` in a way that converts :math:`\mathbf{P}` from *world space* to *measurement space* (we'll learn what that means). It is nearly impossible to see that the Kalman gain (`K`) is just a ratio of one number divided by a second number which has been converted to a different basis. This statement may not convey a lot of information to you before reading the book, but I assure you that :math:`\mathbf{K} = \mathbf{PH}^\mathsf{T}[\mathbf{HPH}^\mathsf{T} + \mathbf{R}]^{-1}` is saying something very succinctly. There are two key pieces of information here - we are finding a ratio, and we are doing it in measurement space. I can see that in my first Python line, I cannot see that in the second line. If you want a counter-argument, my version obscures the information that :math:`\mathbf{P}` is in this context is a *prior* . 

These comments apply to library code. Calling code should use names like `sensor_noise`, or `gps_sensor_noise`, not `R`. Math code should read like math, and interface or glue code should read like normal code. Context is important.

I will not *win* this argument, and some people will not agree with my naming choices. I will finish by stating, very truthfully, that I made two mistakes the first time I typed the second version and it took me awhile to find it. In any case, I aim for using the mathematical symbol names whenever possible, coupled with readable class and function names. So, it is `KalmanFilter.P`, not `KF.P` and not `KalmanFilter.apriori_state_covariance`. 


Communication
=============

Unless it is deeply private (you don't want someone else seeing propietary
code, for example), please ask questions and such on the issue tracker,
not by email. This is solely so that everyone gets to see the answer. "Issue"
doesn't mean bug. 


Modules
=======

filterpy.kalman
---------------

The classes in this submodule implement the various Kalman filters. There is
also support for smoother functions. The smoothers are methods of the classes.
For example, the KalmanFilter class contains rts_smoother to perform 
Rauch-Tung-Striebel smoothing.
   
  
Linear Kalman Filters
+++++++++++++++++++++

Implements various Kalman filters using the linear equations form of the filter.

.. toctree::
    :maxdepth: 1

    kalman/KalmanFilter
    kalman/Saver
    kalman/FixedLagSmoother
    kalman/SquareRootFilter
    kalman/InformationFilter
    kalman/FadingKalmanFilter
    kalman/MMAEFilterBank
    kalman/IMMEstimator

   
Extended Kalman Filter
++++++++++++++++++++++
.. toctree::
    :maxdepth: 1

    kalman/ExtendedKalmanFilter
   
Unscented Kalman Filter
+++++++++++++++++++++++
These modules are used to implement the Unscented Kalman filter.

.. toctree::
    :maxdepth: 1
   
    kalman/UnscentedKalmanFilter
    kalman/unscented_transform
 
Ensemble Kalman Filter
+++++++++++++++++++++++
.. toctree::
    :maxdepth: 1

    kalman/EnsembleKalmanFilter


filterpy.common
---------------

Contains various useful functions that support the filtering classes 
and functions. Most useful are functions to compute the process noise
matrix Q. It also implements the Van Loan discretization of a linear
differential equation.

.. toctree::
    :maxdepth: 1
   
    common/common


filterpy.stats
--------------

Contains statistical functions useful for Kalman filtering such as
multivariate Gaussian multiplication, computing the log-likelihood,
NESS, and mahalanobis distance, along with plotting routines to plot
multivariate Gaussians CDFs, PDFs, and covariance ellipses.

.. toctree::
    :maxdepth: 1
   
    stats/stats


filterpy.monte_carlo
--------------------

Routines for Markov Chain Monte Carlo (MCMC) computation, mainly for
particle filtering.

.. toctree::
    :maxdepth: 1
   
    monte_carlo/resampling

filterpy.discrete_bayes
-----------------------

Routines for performing discrete Bayes filtering.

.. toctree::
    :maxdepth: 1
   
    discrete_bayes/discrete_bayes

filterpy.gh
-----------

These classes various g-h filters. The functions are helpers that provide
settings for the *g* and *h* parameters for various common filters.

.. toctree::
    :maxdepth: 1
   
    gh/GHFilterOrder
    gh/GHFilter
    gh/GHKFilter
    gh/optimal_noise_smoothing
    gh/least_squares_parameters
    gh/critical_damping_parameters
    gh/benedict_bornder_constants



filterpy.memory
---------------

Implements a polynomial fading memory filter. You can achieve the same
results, and more, using the KalmanFilter class. However, some books
use this form of the fading memory filter, so it is here for completeness.
I suppose some would also find this simpler to use than the standard
Kalman filter.

.. toctree::
    :maxdepth: 1
   
    memory/FadingMemoryFilter


filterpy.hinfinity
------------------

.. toctree::
    :maxdepth: 1
   
    hinfinity/HInfinityFilter

filterpy.leastsq
++++++++++++++++

.. toctree::
    :maxdepth: 1
   
    leastsq/LeastSquaresFilter

   
References
==========


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
