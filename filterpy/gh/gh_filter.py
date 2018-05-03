# -*- coding: utf-8 -*-
# pylint: disable=C0103, R0913, R0902, C0326, R0903, W1401, too-many-lines
# disable snake_case warning, too many arguments, too many attributes,
# one space before assignment, too few public methods, anomalous backslash
# in string

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import dot
from filterpy.common import pretty_str


class GHFilterOrder(object):
    """ A g-h filter of aspecified order 0, 1, or 2.

    Strictly speaking, the g-h filter is order 1, and the 2nd order
    filter is called the g-h-k filter. I'm not aware of any filter name
    that encompasses orders 0, 1, and 2 under one name, or I would use it.


    Parameters
    ----------

    x0 : 1D np.array or scalar
        Initial value for the filter state. Each value can be a scalar
        or a np.array.

        You can use a scalar for x0. If order > 0, then 0.0 is assumed
        for the higher order terms.

        x[0] is the value being tracked
        x[1] is the first derivative (for order 1 and 2 filters)
        x[2] is the second derivative (for order 2 filters)

    dt : scalar
        timestep

    order : int
        order of the filter. Defines the order of the system
        0 - assumes system of form x = a_0 + a_1*t
        1 - assumes system of form x = a_0 +a_1*t + a_2*t^2
        2 - assumes system of form x = a_0 +a_1*t + a_2*t^2 + a_3*t^3

    g : float
        filter g gain parameter.

    h : float, optional
        filter h gain parameter, order 1 and 2 only

    k : float, optional
        filter k gain parameter, order 2 only

    Atrributes
    -------

    x : np.array
        State of the filter.

        x[0] is the value being tracked
        x[1] is the derivative of x[0] (order 1 and 2 only)
        x[2] is the 2nd derivative of x[0] (order 2 only)

        This is always an np.array, even for order 0 where you can
        initialize x0 with a scalar.

    y : np.array
        Residual - difference between the measurement and the prediction

    dt : scalar
        timestep

    order : int
        order of the filter. Defines the order of the system
        0 - assumes system of form x = a_0 + a_1*t
        1 - assumes system of form x = a_0 +a_1*t + a_2*t^2
        2 - assumes system of form x = a_0 +a_1*t + a_2*t^2 + a_3*t^3

    g : float
        filter g gain parameter.

    h : float
        filter h gain parameter, order 1 and 2 only

    k : float
        filter k gain parameter, order 2 only

    z : 1D np.array or scalar
        measurement passed into update()

    """


    def __init__(self, x0, dt, order, g, h=None, k=None):
        """ Creates a g-h filter of order 0, 1, or 2.

        """

        if order < 0 or order > 2:
            raise ValueError('order must be between 0 and 2')

        if np.isscalar(x0):
            self.x = np.zeros(order+1)
            self.x[0] = x0
        else:
            self.x = np.copy(x0.astype(float))

        self.dt = dt
        self.order = order

        self.g = g
        self.h = h
        self.k = k
        self.y = np.zeros(len(self.x)) # residual
        self.z = np.zeros(len(self.x)) # last measurement



    def update(self, z, g=None, h=None, k=None):
        """
        Update the filter with measurement z. z must be the same type
        or treatable as the same type as self.x[0].
        """

        if self.order == 0:
            if g is None:
                g = self.g
            self.y = z - self.x[0]
            self.x += dot(g, self.y)

        elif self.order == 1:
            if g is None:
                g = self.g
            if h is None:
                h = self.h
            x  = self.x[0]
            dx = self.x[1]
            dxdt = dot(dx, self.dt)

            self.y = z - (x + dxdt)
            self.x[0] = x + dxdt + g*self.y
            self.x[1] = dx       + h*self.y / self.dt

            self.z = z

        else: # order == 2
            if g is None:
                g = self.g
            if h is None:
                h = self.h
            if k is None:
                k = self.k

            x   = self.x[0]
            dx  = self.x[1]
            ddx = self.x[2]
            dxdt = dot(dx, self.dt)
            T2 = self.dt**2.

            self.y = z -(x + dxdt +0.5*ddx*T2)

            self.x[0] = x + dxdt + 0.5*ddx*T2 + g*self.y
            self.x[1] = dx + ddx*self.dt      + h*self.y / self.dt
            self.x[2] = ddx                 + 2*k*self.y / (self.dt**2)

    def __repr__(self):
        return '\n'.join([
            'GHFilterOrder object',
            pretty_str('dt', self.dt),
            pretty_str('order', self.order),
            pretty_str('x', self.x),
            pretty_str('g', self.g),
            pretty_str('h', self.h),
            pretty_str('k', self.k),
            pretty_str('y', self.y),
            pretty_str('z', self.z)
            ])


class GHFilter(object):
    """
    Implements the g-h filter. The topic is too large to cover in
    this comment. See my book "Kalman and Bayesian Filters in Python" [1]
    or Eli Brookner's "Tracking and Kalman Filters Made Easy" [2].

    A few basic examples are below, and the tests in ./gh_tests.py may
    give you more ideas on use.


    Parameters
    ----------

    x : 1D np.array or scalar
        Initial value for the filter state. Each value can be a scalar
        or a np.array.

        You can use a scalar for x0. If order > 0, then 0.0 is assumed
        for the higher order terms.

        x[0] is the value being tracked
        x[1] is the first derivative (for order 1 and 2 filters)
        x[2] is the second derivative (for order 2 filters)

    dx : 1D np.array or scalar
        Initial value for the derivative of the filter state.

    dt : scalar
        time step

    g : float
        filter g gain parameter.

    h : float
        filter h gain parameter.


    Attributes
    ----------
    x : 1D np.array or scalar
        filter state

    dx : 1D np.array or scalar
       derivative of the filter state.

    x_prediction : 1D np.array or scalar
        predicted filter state

    dx_prediction : 1D np.array or scalar
       predicted derivative of the filter state.

    dt : scalar
        time step

    g : float
        filter g gain parameter.

    h : float
        filter h gain parameter.

    y : np.array, or scalar
        residual (difference between measurement and prior)

    z : np.array, or scalar
        measurement passed into update()

    Examples
    --------

    Create a basic filter for a scalar value with g=.8, h=.2.
    Initialize to 0, with a derivative(velocity) of 0.

    >>> from filterpy.gh import GHFilter
    >>> f = GHFilter (x=0., dx=0., dt=1., g=.8, h=.2)

    Incorporate the measurement of 1

    >>> f.update(z=1)
    (0.8, 0.2)

    Incorporate a measurement of 2 with g=1 and h=0.01

    >>> f.update(z=2, g=1, h=0.01)
    (2.0, 0.21000000000000002)

    Create a filter with two independent variables.

    >>> from numpy import array
    >>> f = GHFilter (x=array([0,1]), dx=array([0,0]), dt=1, g=.8, h=.02)

    and update with the measurements (2,4)

    >>> f.update(array([2,4])
    (array([ 1.6,  3.4]), array([ 0.04,  0.06]))


    References
    ----------

    [1] Labbe, "Kalman and Bayesian Filters in Python"
    http://rlabbe.github.io/Kalman-and-Bayesian-Filters-in-Python

    [2] Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and
    Sons, 1998.

    """


    def __init__(self, x, dx, dt, g, h):
        self.x = x
        self.dx = dx
        self.dt = dt
        self.g = g
        self.h = h
        self.dx_prediction = self.dx
        self.x_prediction  = self.x

        if np.ndim(x) == 0:
            self.y = 0.   # residual
            self.z = 0.
        else:
            self.y = np.zeros(len(x))
            self.z = np.zeros(len(x))


    def update(self, z, g=None, h=None):
        """
        performs the g-h filter predict and update step on the
        measurement z. Modifies the member variables listed below,
        and returns the state of x and dx as a tuple as a convienence.

        **Modified Members**

        x
            filtered state variable

        dx
            derivative (velocity) of x

        residual
            difference between the measurement and the prediction for x

        x_prediction
            predicted value of x before incorporating the measurement z.

        dx_prediction
            predicted value of the derivative of x before incorporating the
            measurement z.

        Parameters
        ----------

        z : any
            the measurement
        g : scalar (optional)
            Override the fixed self.g value for this update
        h : scalar (optional)
            Override the fixed self.h value for this update

        Returns
        -------

        x filter output for x
        dx filter output for dx (derivative of x
        """

        if g is None:
            g = self.g
        if h is None:
            h = self.h

        #prediction step
        self.dx_prediction = self.dx
        self.x_prediction  = self.x + (self.dx*self.dt)

        # update step
        self.y = z - self.x_prediction
        self.dx = self.dx_prediction + h * self.y / self.dt
        self.x  = self.x_prediction  + g * self.y

        return (self.x, self.dx)


    def batch_filter(self, data, save_predictions=False, saver=None):
        """
        Given a sequenced list of data, performs g-h filter
        with a fixed g and h. See update() if you need to vary g and/or h.

        Uses self.x and self.dx to initialize the filter, but DOES NOT
        alter self.x and self.dx during execution, allowing you to use this
        class multiple times without reseting self.x and self.dx. I'm not sure
        how often you would need to do that, but the capability is there.
        More exactly, none of the class member variables are modified
        by this function, in distinct contrast to update(), which changes
        most of them.

        Parameters
        ----------

        data : list like
            contains the data to be filtered.

        save_predictions : boolean
            the predictions will be saved and returned if this is true

        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch


        Returns
        -------

        results : np.array shape (n+1, 2), where n=len(data)
           contains the results of the filter, where
           results[i,0] is x , and
           results[i,1] is dx (derivative of x)
           First entry is the initial values of x and dx as set by __init__.

        predictions : np.array shape(n), optional
           the predictions for each step in the filter. Only retured if
           save_predictions == True
        """

        x = self.x
        dx = self.dx
        n = len(data)

        results = np.zeros((n+1, 2))
        results[0, 0] = x
        results[0, 1] = dx

        if save_predictions:
            predictions = np.zeros(n)

        # optimization to avoid n computations of h / dt
        h_dt = self.h / self.dt

        for i, z in enumerate(data):
            #prediction step
            x_est = x + (dx * self.dt)

            # update step
            residual = z - x_est
            dx = dx    + h_dt   * residual # i.e. dx = dx + h * residual / dt
            x  = x_est + self.g * residual

            results[i+1, 0] = x
            results[i+1, 1] = dx
            if save_predictions:
                predictions[i] = x_est

            if saver is not None:
                saver.save()

        if save_predictions:
            return results, predictions

        return results


    def VRF_prediction(self):
        """
        Returns the Variance Reduction Factor of the prediction
        step of the filter. The VRF is the
        normalized variance for the filter, as given in the equation below.

        .. math::
            VRF(\hat{x}_{n+1,n}) = \\frac{VAR(\hat{x}_{n+1,n})}{\sigma^2_x}

        References
        ----------

        Asquith, "Weight Selection in First Order Linear Filters"
        Report No RG-TR-69-12, U.S. Army Missle Command. Redstone Arsenal, Al.
        November 24, 1970.
        """

        g = self.g
        h = self.h

        return (2*g**2 + 2*h + g*h) / (g*(4 - 2*g - h))


    def VRF(self):
        """
        Returns the Variance Reduction Factor (VRF) of the state variable
        of the filter (x) and its derivatives (dx, ddx). The VRF is the
        normalized variance for the filter, as given in the equations below.

        .. math::
            VRF(\hat{x}_{n,n}) = \\frac{VAR(\hat{x}_{n,n})}{\sigma^2_x}

            VRF(\hat{\dot{x}}_{n,n}) = \\frac{VAR(\hat{\dot{x}}_{n,n})}{\sigma^2_x}

            VRF(\hat{\ddot{x}}_{n,n}) = \\frac{VAR(\hat{\ddot{x}}_{n,n})}{\sigma^2_x}

        Returns
        -------

        vrf_x   VRF of x state variable
        vrf_dx  VRF of the dx state variable (derivative of x)
        """

        g = self.g
        h = self.h

        den = g*(4 - 2*g - h)

        vx = (2*g**2 + 2*h - 3*g*h) / den
        vdx = 2*h**2 / (self.dt**2 * den)

        return (vx, vdx)

    def __repr__(self):
        return '\n'.join([
            'GHFilter object',
            pretty_str('dt', self.dt),
            pretty_str('g', self.g),
            pretty_str('h', self.h),
            pretty_str('x', self.x),
            pretty_str('dx', self.dx),
            pretty_str('x_prediction', self.x_prediction),
            pretty_str('dx_prediction', self.dx_prediction),
            pretty_str('y', self.y),
            pretty_str('z', self.z)
            ])


class GHKFilter(object):
    """
    Implements the g-h-k filter.

    Parameters
    ----------

    x : 1D np.array or scalar
        Initial value for the filter state. Each value can be a scalar
        or a np.array.

        You can use a scalar for x0. If order > 0, then 0.0 is assumed
        for the higher order terms.

        x[0] is the value being tracked
        x[1] is the first derivative (for order 1 and 2 filters)
        x[2] is the second derivative (for order 2 filters)

    dx : 1D np.array or scalar
        Initial value for the derivative of the filter state.

    ddx : 1D np.array or scalar
        Initial value for the second derivative of the filter state.

    dt : scalar
        time step

    g : float
        filter g gain parameter.

    h : float
        filter h gain parameter.

    k : float
        filter k gain parameter.



    Attributes
    ----------
    x : 1D np.array or scalar
        filter state

    dx : 1D np.array or scalar
       derivative of the filter state.

    ddx : 1D np.array or scalar
       second derivative of the filter state.

    x_prediction : 1D np.array or scalar
        predicted filter state

    dx_prediction : 1D np.array or scalar
       predicted derivative of the filter state.

    ddx_prediction : 1D np.array or scalar
       second predicted derivative of the filter state.

    dt : scalar
        time step

    g : float
        filter g gain parameter.

    h : float
        filter h gain parameter.

    k : float
        filter k gain parameter.

    y : np.array, or scalar
        residual (difference between measurement and prior)

    z : np.array, or scalar
        measurement passed into update()

    References
    ----------

    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and
    Sons, 1998.
    """

    def __init__(self, x, dx, ddx, dt, g, h, k):
        self.x = x
        self.dx = dx
        self.ddx = ddx
        self.x_prediction = self.x
        self.dx_prediction = self.dx
        self.ddx_prediction = self.ddx

        self.dt = dt
        self.g = g
        self.h = h
        self.k = k

        if np.ndim(x) == 0:
            self.y = 0.  # residual
            self.z = 0.
        else:
            self.y = np.zeros(len(x))
            self.z = np.zeros(len(x))


    def update(self, z, g=None, h=None, k=None):
        """
        Performs the g-h filter predict and update step on the
        measurement z.

        On return, self.x, self.dx, self.y, and self.x_prediction
        will have been updated with the results of the computation. For
        convienence, self.x and self.dx are returned in a tuple.

        Parameters
        ----------

        z : scalar
            the measurement
        g : scalar (optional)
            Override the fixed self.g value for this update
        h : scalar (optional)
            Override the fixed self.h value for this update
        k : scalar (optional)
            Override the fixed self.k value for this update

        Returns
        -------

        x filter output for x
        dx filter output for dx (derivative of x

        """

        if g is None:
            g = self.g
        if h is None:
            h = self.h
        if k is None:
            k = self.k

        dt = self.dt
        dt_sqr = dt**2
        #prediction step
        self.ddx_prediction = self.ddx
        self.dx_prediction  = self.dx + self.ddx*dt
        self.x_prediction   = self.x  + self.dx*dt + .5*self.ddx*(dt_sqr)

        # update step
        self.y = z - self.x_prediction

        self.ddx = self.ddx_prediction + 2*k*self.y / dt_sqr
        self.dx  = self.dx_prediction  + h * self.y / dt
        self.x   = self.x_prediction   + g * self.y

        return (self.x, self.dx)


    def batch_filter(self, data, save_predictions=False):
        """
        Performs g-h filter with a fixed g and h.

        Uses self.x and self.dx to initialize the filter, but DOES NOT
        alter self.x and self.dx during execution, allowing you to use this
        class multiple times without reseting self.x and self.dx. I'm not sure
        how often you would need to do that, but the capability is there.
        More exactly, none of the class member variables are modified
        by this function.

        Parameters
        ----------

        data : list_like
            contains the data to be filtered.

        save_predictions : boolean
            The predictions will be saved and returned if this is true

        Returns
        -------

        results : np.array shape (n+1, 2), where n=len(data)
           contains the results of the filter, where
           results[i,0] is x , and
           results[i,1] is dx (derivative of x)
           First entry is the initial values of x and dx as set by __init__.

        predictions : np.array shape(n), or None
           the predictions for each step in the filter. Only returned if
           save_predictions == True
        """

        x = self.x
        dx = self.dx
        n = len(data)

        results = np.zeros((n+1, 2))
        results[0, 0] = x
        results[0, 1] = dx

        if save_predictions:
            predictions = np.zeros(n)

        # optimization to avoid n computations of h / dt
        h_dt = self.h / self.dt

        for i, z in enumerate(data):
            #prediction step
            x_est = x + (dx*self.dt)

            # update step
            residual = z - x_est
            dx = dx    + h_dt   * residual # i.e. dx = dx + h * residual / dt
            x  = x_est + self.g * residual

            results[i+1, 0] = x
            results[i+1, 1] = dx
            if save_predictions:
                predictions[i] = x_est

        if save_predictions:
            return results, predictions

        return results


    def VRF_prediction(self):
        """
        Returns the Variance Reduction Factor for x of the prediction
        step of the filter.

        This implements the equation

        .. math::
            VRF(\hat{x}_{n+1,n}) = \\frac{VAR(\hat{x}_{n+1,n})}{\sigma^2_x}

        References
        ----------

        Asquith and Woods, "Total Error Minimization in First
        and Second Order Prediction Filters" Report No RE-TR-70-17, U.S.
        Army Missle Command. Redstone Arsenal, Al. November 24, 1970.
        """

        g = self.g
        h = self.h
        k = self.k
        gh2 = 2*g + h
        return ((g*k*(gh2-4)+ h*(g*gh2+2*h)) /
                (2*k - (g*(h+k)*(gh2-4))))


    def bias_error(self, dddx):
        """
        Returns the bias error given the specified constant jerk(dddx)

        Parameters
        ----------

        dddx : type(self.x)
            3rd derivative (jerk) of the state variable x.

        References
        ----------

        Asquith and Woods, "Total Error Minimization in First
        and Second Order Prediction Filters" Report No RE-TR-70-17, U.S.
        Army Missle Command. Redstone Arsenal, Al. November 24, 1970.
        """
        return -self.dt**3 * dddx / (2*self.k)


    def VRF(self):
        """
        Returns the Variance Reduction Factor (VRF) of the state variable
        of the filter (x) and its derivatives (dx, ddx). The VRF is the
        normalized variance for the filter, as given in the equations below.

        .. math::
            VRF(\hat{x}_{n,n}) = \\frac{VAR(\hat{x}_{n,n})}{\sigma^2_x}

            VRF(\hat{\dot{x}}_{n,n}) = \\frac{VAR(\hat{\dot{x}}_{n,n})}{\sigma^2_x}

            VRF(\hat{\ddot{x}}_{n,n}) = \\frac{VAR(\hat{\ddot{x}}_{n,n})}{\sigma^2_x}

        Returns
        -------

        vrf_x : type(x)
            VRF of x state variable

        vrf_dx : type(x)
            VRF of the dx state variable (derivative of x)

        vrf_ddx : type(x)
            VRF of the ddx state variable (second derivative of x)
        """

        g = self.g
        h = self.h
        k = self.k

        # common subexpressions in the equations pulled out for efficiency,
        # they don't 'mean' anything.
        hg4 = 4- 2*g - h
        ghk = g*h + g*k - 2*k

        vx = (2*h*(2*(g**2) + 2*h - 3*g*h) - 2*g*k*hg4) / (2*k - g*(h+k) * hg4)
        vdx = (2*(h**3) - 4*(h**2)*k + 4*(k**2)*(2-g)) / (2*hg4*ghk)
        vddx = 8*h*(k**2) / ((self.dt**4)*hg4*ghk)

        return (vx, vdx, vddx)


    def __repr__(self):
        return '\n'.join([
            'GHFilter object',
            pretty_str('dt', self.dt),
            pretty_str('g', self.g),
            pretty_str('h', self.h),
            pretty_str('k', self.k),
            pretty_str('x', self.x),
            pretty_str('dx', self.dx),
            pretty_str('ddx', self.ddx),
            pretty_str('x_prediction', self.x_prediction),
            pretty_str('dx_prediction', self.dx_prediction),
            pretty_str('ddx_prediction', self.dx_prediction),
            pretty_str('y', self.y),
            pretty_str('z', self.z)
            ])


def optimal_noise_smoothing(g):
    """ provides g,h,k parameters for optimal smoothing of noise for a given
    value of g. This is due to Polge and Bhagavan[1].

    Parameters
    ----------

    g : float
        value for g for which we will optimize for

    Returns
    -------

    (g,h,k) : (float, float, float)
        values for g,h,k that provide optimal smoothing of noise


    Examples
    --------

    .. code-block:: Python

        from filterpy.gh import GHKFilter, optimal_noise_smoothing

        g,h,k = optimal_noise_smoothing(g)
        f = GHKFilter(0,0,0,1,g,h,k)
        f.update(1.)


    References
    ----------

    [1] Polge and Bhagavan. "A Study of the g-h-k Tracking Filter".
    Report No. RE-CR-76-1. University of Alabama in Huntsville.
    July, 1975

    """

    h = ((2*g**3 - 4*g**2) + (4*g**6 -64*g**5 + 64*g**4)**.5) / (8*(1-g))
    k = (h*(2-g) - g**2) / g

    return (g, h, k)


def least_squares_parameters(n):
    """ An order 1 least squared filter can be computed by a g-h filter
    by varying g and h over time according to the formulas below, where
    the first measurement is at n=0, the second is at n=1, and so on:

    .. math::

        h_n = \\frac{6}{(n+2)(n+1)}

        g_n = \\frac{2(2n+1)}{(n+2)(n+1)}

    Parameters
    ----------

    n : int
        the nth measurement, starting at 0 (i.e. first measurement has n==0)

    Returns
    -------

    (g,h)  : (float, float)
        g and h parameters for this time step for the least-squares filter

    Examples
    --------

    .. code-block:: Python

        from filterpy.gh import GHFilter, least_squares_parameters

        lsf = GHFilter (0, 0, 1, 0, 0)
        z = 10
        for i in range(10):
            g,h = least_squares_parameters(i)
            lsf.update(z, g, h)

    """
    den = (n+2)*(n+1)

    g = (2*(2*n + 1)) / den
    h = 6 / den
    return (g, h)


def critical_damping_parameters(theta, order=2):
    """ Computes values for g and h (and k for g-h-k filter) for a
    critically damped filter.

    The idea here is to create a filter that reduces the influence of
    old data as new data comes in. This allows the filter to track a
    moving target better. This goes by different names. It may be called the
    discounted least-squares g-h filter, a fading-memory polynomal filter
    of order 1, or a critically damped g-h filter.

    In a normal least-squares filter we compute the error for each point as

    .. math::

        \epsilon_t = (z-\\hat{x})^2

    For a crically damped filter we reduce the influence of each error by

     .. math::

         \\theta^{t-i}

    where

     .. math::

         0 <= \\theta <= 1

    In other words the last error is scaled by theta, the next to last by
    theta squared, the next by theta cubed, and so on.

    Parameters
    ----------

    theta : float, 0 <= theta <= 1
        scaling factor for previous terms

    order : int, 2 (default) or 3
       order of filter to create the parameters for. g and h will be
       calculated for the order 2, and g, h, and k for order 3.

    Returns
    -------
    g : scalar
        optimal value for g in the g-h or g-h-k filter

    h : scalar
        optimal value for h in the g-h or g-h-k filter

    k : scalar
        optimal value for g in the g-h-k filter

    Examples
    --------

    .. code-block:: Python

        from filterpy.gh import GHFilter, critical_damping_parameters

        g,h = critical_damping_parameters(0.3)
        critical_filter = GHFilter(0, 0, 1, g, h)

    References
    ----------

    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and
    Sons, 1998.

    Polge and Bhagavan. "A Study of the g-h-k Tracking Filter".
    Report No. RE-CR-76-1. University of Alabama in Huntsville.
    July, 1975

    """
    if theta < 0 or theta > 1:
        raise ValueError('theta must be between 0 and 1')

    if order == 2:
        return (1. - theta**2, (1. - theta)**2)

    if order == 3:
        return (1. - theta**3, 1.5*(1.-theta**2)*(1.-theta), .5*(1 - theta)**3)

    raise ValueError('bad order specified: {}'.format(order))


def benedict_bornder_constants(g, critical=False):
    """ Computes the g,h constants for a Benedict-Bordner filter, which
    minimizes transient errors for a g-h filter.

    Returns the values g,h for a specified g. Strictly speaking, only h
    is computed, g is returned unchanged.

    The default formula for the Benedict-Bordner allows ringing. We can
    "nearly" critically damp it; ringing will be reduced, but not entirely
    eliminated at the cost of reduced performance.

    Parameters
    ----------

    g : float
        scaling factor g for the filter

    critical : boolean, default False
        Attempts to critically damp the filter.

    Returns
    -------

    g : float
        scaling factor g (same as the g that was passed in)

    h : float
        scaling factor h that minimizes the transient errors

    Examples
    --------

    .. code-block:: Python

        from filterpy.gh import GHFilter, benedict_bornder_constants
        g, h = benedict_bornder_constants(.855)
        f = GHFilter(0, 0, 1, g, h)

    References
    ----------

    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and
    Sons, 1998.

    """

    g_sqr = g**2
    if critical:
        return (g, 0.8 * (2. - g_sqr - 2*(1-g_sqr)**.5) / g_sqr)

    return (g, g_sqr / (2.-g))
