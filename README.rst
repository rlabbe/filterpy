FilterPy - Kalman filters and other optimal and non-optimal estimation filters in Python.
-----------------------------------------------------------------------------------------

.. image:: https://img.shields.io/pypi/v/filterpy.svg
        :target: https://pypi.python.org/pypi/filterpy
        
.. image:: https://readthedocs.org/projects/pip/badge/?version=latest&style=flat
        :target: https://filterpy.readthedocs.io/en/latest/        
     

**NOTE**: Imminent drop of support of Python 2.7, 3.4. See section below for details.

This library provides Kalman filtering and various related optimal and
non-optimal filtering software written in Python. It contains Kalman
filters, Extended Kalman filters, Unscented Kalman filters, Kalman
smoothers, Least Squares filters, fading memory filters, g-h filters,
discrete Bayes, and more.

This is code I am developing in conjunction with my book Kalman and
Bayesian Filter in Python, which you can read/download at
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/

My aim is largely pedalogical - I opt for clear code that matches the
equations in the relevant texts on a 1-to-1 basis, even when that has a
performance cost. There are places where this tradeoff is unclear - for
example, I find it somewhat clearer to write a small set of equations
using linear algebra, but numpy's overhead on small matrices makes it
run slower than writing each equation out by hand. Furthermore, books
such Zarchan present the written out form, not the linear algebra form.
It is hard for me to choose which presentation is 'clearer' - it depends
on the audience. In that case I usually opt for the faster implementation.

I use NumPy and SciPy for all of the computations. I have experimented
with Numba and it yields impressive speed ups with minimal costs, but I 
am not convinced that I want to add that requirement to my project. It 
is still on my list of things to figure out, however.

Sphinx generated documentation lives at http://filterpy.readthedocs.org/.
Generation is triggered by git when I do a check in, so this will always
be bleeding edge development version - it will often be ahead of the
released version. 


Plan for dropping Python 2.7 support
------------------------------------

I haven't finalized my decision on this, but NumPy is dropping
Python 2.7 support in December 2018. I will certainly drop Python
2.7 support by then; I will probably do it much sooner.

At the moment FilterPy is on version 1.x. I plan to fork the project
to version 2.0, and support only Python 3.5+. The 1.x version 
will still be available, but I will not support it. If I add something
amazing to 2.0 and someone really begs, I might backport it; more
likely I would accept a pull request with the feature backported
to 1.x. But to be honest I don't forsee this happening.

Why 3.5+, and not 3.4+? 3.5 introduced the matrix multiply symbol,
and I want my code to take advantage of it. Plus, to be honest,
I'm being selfish. I don't want to spend my life supporting this
package, and moving as far into the present as possible means
a few extra years before the Python version I choose becomes
hopelessly dated and a liability. I recognize this makes people
running the default Python in their linux distribution more
painful. All I can say is I did not decide to do the Python
3 fork, and I don't have the time to support the bifurcation
any longer.

I am making edits to the package now in support of my book;
once those are done I'll probably create the 2.0 branch. 
I'm contemplating a SLAM addition to the book, and am not
sure if I will do this in 3.5+ only or not.


Installation
------------

The most general installation is just to use pip, which should come with
any modern Python distribution.

.. image:: https://img.shields.io/pypi/v/filterpy.svg
        :target: https://pypi.python.org/pypi/filterpy
        
::

    pip install filterpy

If you prefer to download the source yourself

::

    cd <directory you want to install to>
    git clone http://github.com/rlabbe/filterpy
    python setup.py install

If you use Anaconda, you can install from the conda-forge channel. You
will need to add the conda-forge channel if you haven't already done so:

::
    conda config --add channels conda-forge
    
and then install with:

::
    conda install filterpy
    
    
And, if you want to install from the bleeding edge git version

::

    pip install git+https://github.com/rlabbe/filterpy.git

Note: I make no guarantees that everything works if you install from here.
I'm the only developer, and so I don't worry about dev/release branches and
the like. Unless I fix a bug for you and tell you to get this version because
I haven't made a new release yet, I strongly advise not installing from git.


    

Basic use
---------

Full documentation is at
https://filterpy.readthedocs.io/en/latest/


First, import the filters and helper functions.

.. code-block:: python

    import numpy as np
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise

Now, create the filter

.. code-block:: python

    my_filter = KalmanFilter(dim_x=2, dim_z=1)


Initialize the filter's matrices.

.. code-block:: python

    my_filter.x = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    my_filter.F = np.array([[1.,1.],
                    [0.,1.]])    # state transition matrix

    my_filter.H = np.array([[1.,0.]])    # Measurement function
    my_filter.P *= 1000.                 # covariance matrix
    my_filter.R = 5                      # state uncertainty
    my_filter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.1) # process uncertainty


Finally, run the filter.

.. code-block:: python

    while True:
        my_filter.predict()
        my_filter.update(get_some_measurement())

        # do something with the output
        x = my_filter.x
        do_something_amazing(x)

Sorry, that is the extent of the documentation here. However, the library
is broken up into subdirectories: gh, kalman, memory, leastsq, and so on.
Each subdirectory contains python files relating to that form of filter.
The functions and methods contain pretty good docstrings on use.

My book https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/
uses this library, and is the place to go if you are trying to learn
about Kalman filtering and/or this library. These two are not exactly in 
sync - my normal development cycle is to add files here, test them, figure 
out how to present them pedalogically, then write the appropriate section
or chapter in the book. So there is code here that is not discussed
yet in the book.


Requirements
------------

This library uses NumPy, SciPy, Matplotlib, and Python. 

I haven't extensively tested backwards compatibility - I use the
Anaconda distribution, and so I am on Python 3.6 and 2.7.14, along with
whatever version of NumPy, SciPy, and matplotlib they provide. But I am
using pretty basic Python - numpy.array, maybe a list comprehension in
my tests.

I import from **__future__** to ensure the code works in Python 2 and 3.


Testing
-------

All tests are written to work with py.test. Just type ``py.test`` at the
command line.

As explained above, the tests are not robust. I'm still at the stage
where visual plots are the best way to see how things are working.
Apologies, but I think it is a sound choice for development. It is easy
for a filter to perform within theoretical limits (which we can write a
non-visual test for) yet be 'off' in some way. The code itself contains
tests in the form of asserts and properties that ensure that arrays are
of the proper dimension, etc.

References
----------

I use three main texts as my refererence, though I do own the majority
of the Kalman filtering literature. First is Paul Zarchan's
'Fundamentals of Kalman Filtering: A Practical Approach'. I think it by
far the best Kalman filtering book out there if you are interested in
practical applications more than writing a thesis. The second book I use
is Eli Brookner's 'Tracking and Kalman Filtering Made Easy'. This is an
astonishingly good book; its first chapter is actually readable by the
layperson! Brookner starts from the g-h filter, and shows how all other
filters - the Kalman filter, least squares, fading memory, etc., all
derive from the g-h filter. It greatly simplifies many aspects of
analysis and/or intuitive understanding of your problem. In contrast,
Zarchan starts from least squares, and then moves on to Kalman
filtering. I find that he downplays the predict-update aspect of the
algorithms, but he has a wealth of worked examples and comparisons
between different methods. I think both viewpoints are needed, and so I
can't imagine discarding one book. Brookner also focuses on issues that
are ignored in other books - track initialization, detecting and
discarding noise, tracking multiple objects, an so on.

I said three books. I also like and use Bar-Shalom's Estimation with
Applications to Tracking and Navigation. Much more mathematical than the
previous two books, I would not recommend it as a first text unless you
already have a background in control theory or optimal estimation. Once
you have that experience, this book is a gem. Every sentence is crystal
clear, his language is precise, but each abstract mathematical statement
is followed with something like "and this means...".


License
-------
.. image:: https://anaconda.org/rlabbe/filterpy/badges/license.svg   :target: https://anaconda.org/rlabbe/filterpy

The MIT License (MIT)

Copyright (c) 2015 Roger R. Labbe Jr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.TION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
