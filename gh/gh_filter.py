# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 18:26:28 2014

@author: rlabbe
"""


from __future__ import division

import numpy as np


class GHFilter(object):
    """ Implements the g-h filter. The topic is too large to cover in
    this comment. See my book "Kalman and Bayesian Filters in Python"
    or Eli Brookner's "Tracking and Kalman Filters Made Easy".
    
    A few basic examples are below, and the tests in ./gh_tests.py may
    give you more ideas on use.
    
    Examples
    --------
    Create a basic filter for a scalar value with g=.8, h=.2.
    Initialize to 0, with a derivative(velocity) of 0.
    
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
    Labbe, "Kalman and Bayesian Filters in Python"
    http://rlabbe.github.io/Kalman-and-Bayesian-Filters-in-Python
    
    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and 
    Sons, 1998.        
    """    
    def __init__(self, x, dx, dt, g, h):
        
        assert np.isscalar(dt)
        assert np.isscalar(g)
        assert np.isscalar(h)
        
        self.x = x
        self.dx = dx            
        self.dt = dt
        self.g = g
        self.h = h
        
        
    def update (self, z, g=None, h=None):
        """performs the g-h filter predict and update step on the 
        measurement z. Modifies the member variables listed below,
        and returns the state of x and dx as a tuple as a convienence.
        
        Modified Members
        ----------------
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
        self.residual = z - self.x_prediction
        self.dx = self.dx_prediction + h * self.residual / self.dt
        self.x  = self.x_prediction  + g * self.residual
        
        return (self.x, self.dx)    



    def batch_filter (self, data, save_predictions=False):
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
        results[0,0] = x
        results[0,1] = dx
        
        if save_predictions:
            predictions = np.zeros(n)
        
        # optimization to avoid n computations of h / dt
        h_dt = self.h / self.dt

        for i,z in enumerate(data):
            #prediction step
            x_est = x + (dx*self.dt)
            
            # update step
            residual = z - x_est
            dx = dx    + h_dt   * residual # i.e. dx = dx + h * residual / dt
            x  = x_est + self.g * residual
            
            results[i+1,0] = x
            results[i+1,1] = dx
            if save_predictions:
                predictions[i] = x_est

        if save_predictions:
            return results, predictions
        else:
            return results


    def VRF_prediction(self):
        """ Returns the Variance Reduction Factor of the prediction
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
        """ Returns the Variance Reduction Factor (VRF) of the state variable 
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


class GHKFilter(object):
    """ Implements the g-h-k filter.
    
    References
    ----------
    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and 
    Sons, 1998.        
    """
    
    def __init__(self, x, dx, ddx, dt, g, h, k):
        
        assert np.isscalar(dt)
        assert np.isscalar(g)
        assert np.isscalar(h)
        
        self.x = x
        self.dx = dx
        self.ddx = ddx         
        self.dt = dt
        self.g = g
        self.h = h
        self.k = k
        
        
    def update (self, z, g=None, h=None, k=None):
        """performs the g-h filter predict and update step on the 
        measurement z.
        
        On return, self.x, self.dx, self.residual, and self.x_prediction
        will have been updated with the results of the computation. For
        convienence, self.x and self.dx are returned in a tuple.
        
        Parameters
        ----------
        z the measurement
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
        self.dx_prediction = self.dx + self.ddx*dt
        self.x_prediction  = self.x  + self.dx*dt + .5*self.ddx*(dt_sqr)
            
        # update step
        self.residual = z - self.x_prediction
        
        self.ddx = self.ddx_prediction + 2*k*self.residual / dt_sqr
        self.dx  = self.dx_prediction  + h * self.residual / dt
        self.x   = self.x_prediction   + g * self.residual
        
        return (self.x, self.dx)    


    def batch_filter (self, data, save_predictions=False):
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
        results[0,0] = x
        results[0,1] = dx
        
        if save_predictions:
            predictions = np.zeros(n)
        
        # optimization to avoid n computations of h / dt
        h_dt = self.h / self.dt

        for i,z in enumerate(data):
            #prediction step
            x_est = x + (dx*self.dt)
            
            # update step
            residual = z - x_est
            dx = dx    + h_dt   * residual # i.e. dx = dx + h * residual / dt
            x  = x_est + self.g * residual
            
            results[i+1,0] = x
            results[i+1,1] = dx
            if save_predictions:
                predictions[i] = x_est

        if save_predictions:
            return results, predictions
        else:
            return results

    
    def VRF_prediction(self):
        """ Returns the Variance Reduction Factor for x of the prediction
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
        """ Returns the bias error given the specified constant jerk(dddx)
        
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
        """ Returns the Variance Reduction Factor (VRF) of the state variable 
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