# -*- coding: utf-8 -*-

"""Copyright 2014-2016 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division, unicode_literals)
from filterpy.common import setter, setter_1d, setter_scalar, dot3
from filterpy.stats import logpdf
import math
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import scipy.linalg as linalg


class KalmanFilter(object):
    """ Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; the defaults  will
    not give you a functional filter.

    You will have to set the following attributes after constructing this
    object for the filter to perform properly. Please note that there are
    various checks in place to ensure that you have made everything the
    'correct' size. However, it is possible to provide incorrectly sized
    arrays such that the linear algebra can not perform an operation.
    It can also fail silently - you can end up with matrices of a size that
    allows the linear algebra to work, but are the wrong shape for the problem
    you are trying to solve.

    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        State estimate vector

    P : numpy.array(dim_x, dim_x)
        Covariance matrix

    R : numpy.array(dim_z, dim_z)
        Measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        Process noise matrix

    F : numpy.array()
        State Transition matrix

    H : numpy.array(dim_x, dim_x)
        Measurement function


    You may read the following attributes.

    Attributes
    ----------
    y : numpy.array
        Residual of the update step.

    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step

    S :  numpy.array
        Systen uncertaintly projected to measurement space

    likelihood : scalar
        Likelihood of last measurement update.

    log_likelihood : scalar
        Log likelihood of last measurement update.
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        Parameters
        ----------
        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.
            This is used to set the default size of P, Q, and u
        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.
        dim_u : int (optional)
            size of the control input, if it is being used.
            Default value of 0 indicates it is not used.
        """

        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.B = 0                # control transition matrix
        self.F = 0                # state transition matrix
        self.H = 0                 # Measurement function
        self.R = eye(dim_z)        # state uncertainty
        self._alpha_sq = 1.        # fading memory control
        self.M = 0                 # process-measurement cross correlation

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty

        # identity matrix. Do not alter this.
        self.I = np.eye(dim_x)


    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------
        z : np.array
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        if z is None:
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        P = self.P
        x = self.x

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        Hx = dot(H, x)

        assert shape(Hx) == shape(z) or (shape(Hx) == (1,1) and shape(z) == (1,)), \
               'shape of z should be {}, but it is {}'.format(
               shape(Hx), shape(z))
        self.y = z - Hx

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot3(H, P, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot3(P, H.T, linalg.inv(self.S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - dot(self.K, H)
        self.P = dot3(I_KH, P, I_KH.T) + dot3(self.K, R, self.K.T)

        self.log_likelihood = logpdf(z, dot(H, x), self.S)


    def update_correlated(self, z, R=None, H=None):
        """ Add a new measurement (z) to the Kalman filter assuming that
        process noise and measurement noise are correlated as defined in
        the `self.M` matrix.

        If z is None, nothing is changed.

        Parameters
        ----------
        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array,  or None
            Optionally provide H to override the measurement function for this
            one call, otherwise  self.H will be used.
        """

        if z is None:
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        x = self.x
        P = self.P
        M = self.M

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, x)

        # project system uncertainty into measurement space
        self.S = dot3(H, P, H.T) + dot(H, M) + dot(M.T, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(dot(P, H.T) + M, linalg.inv(self.S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(self.K, self.y)
        self.P = P - dot(self.K, dot(H, P) + M.T)

        # compute log likelihood
        self.log_likelihood = logpdf(z, dot(H, x), self.S)


    def test_matrix_dimensions(self, z=None, H=None, R=None, F=None, Q=None):
        """ Performs a series of asserts to check that the size of everything
        is what it should be. This can help you debug problems in your design.

        If you pass in H, R, F, Q those will be used instead of this object's
        value for those matrices.

        Testing `z` (the measurement) is problamatic. x is a vector, and can be
        implemented as either a 1D array or as a nx1 column vector. Thus Hx
        can be of different shapes. Then, if Hx is a single value, it can
        be either a 1D array or 2D vector. If either is true, z can reasonably
        be a scalar (either '3' or np.array('3') are scalars under this
        definition), a 1D, 1 element array, or a 2D, 1 element array. You are
        allowed to pass in any combination that works.
        """

        if H is None: H = self.H
        if R is None: R = self.R
        if F is None: F = self.F
        if Q is None: Q = self.Q
        x = self.x
        P = self.P

        assert x.ndim == 1 or x.ndim == 2, \
                "x must have one or two dimensions, but has {}".format(
                x.ndim)

        if x.ndim == 1:
            assert x.shape[0] == self.dim_x, \
                   "Shape of x must be ({},{}), but is {}".format(
                   self.dim_x, 1, x.shape)
        else:
            assert x.shape == (self.dim_x, 1), \
                   "Shape of x must be ({},{}), but is {}".format(
                   self.dim_x, 1, x.shape)

        assert P.shape == (self.dim_x, self.dim_x), \
               "Shape of P must be ({},{}), but is {}".format(
               self.dim_x, self.dim_x, P.shape)

        assert Q.shape == (self.dim_x, self.dim_x), \
               "Shape of P must be ({},{}), but is {}".format(
               self.dim_x, self.dim_x, P.shape)

        assert F.shape == (self.dim_x, self.dim_x), \
               "Shape of F must be ({},{}), but is {}".format(
               self.dim_x, self.dim_x, F.shape)


        assert np.ndim(H) == 2, \
               "Shape of H must be (dim_z, {}), but is {}".format(
               P.shape[0], shape(H))

        assert H.shape[1] == P.shape[0], \
               "Shape of H must be (dim_z, {}), but is {}".format(
               P.shape[0], H.shape)

        # shape of R must be the same as HPH'
        hph_shape = (H.shape[0], H.shape[0])
        r_shape = shape(R)

        if H.shape[0] == 1:
            # r can be scalar, 1D, or 2D in this case
            assert r_shape == () or r_shape == (1,) or r_shape == (1,1), \
            "R must be scalar or one element array, but is shaped {}".format(
            r_shape)
        else:
            assert r_shape == hph_shape, \
            "shape of R should be {} but it is {}".format(hph_shape, r_shape)


        if z is not None:
            z_shape = shape(z)
        else:
            z_shape = (self.dim_z, 1)

        # H@x must have shape of z
        Hx = dot(H, x)

        if z_shape == (): # scalar or np.array(scalar)
            assert Hx.ndim == 1 or shape(Hx) == (1,1), \
            "shape of z should be {}, not {} for the given H".format(
                   shape(Hx), z_shape)

        elif shape(Hx) == (1,):
            assert z_shape[0] == 1, 'Shape of z must be {} for the given H'.format(shape(Hx))

        else:
            assert (z_shape == shape(Hx) or
                    (len(z_shape) == 1 and shape(Hx) == (z_shape[0], 1))), \
                    "shape of z should be {}, not {} for the given H".format(
                    shape(Hx), z_shape)

        if np.ndim(Hx) > 1 and shape(Hx) != (1,1):
            assert shape(Hx) == z_shape, \
               'shape of z should be {} for the given H, but it is {}'.format(
               shape(Hx), z_shape)


    def predict(self, u=0, B=None, F=None, Q=None):
        """ Predict next position using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None in
            any position will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None in
            any position will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None in
            any position will cause the filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        self.x = dot(F, self.x) + dot(B, u)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot3(F, self.P, F.T) + Q


    def batch_filter(self, zs, Fs=None, Qs=None, Hs=None, Rs=None, Bs=None, us=None, update_first=False):
        """ Batch processes a sequences of measurements.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self.dt` Missing
            measurements must be represented by 'None'.

        Fs : list-like, optional
            optional list of values to use for the state transition matrix matrix;
            a value of None in any position will cause the filter
            to use `self.F` for that time step. If Fs is None then self.F is
            used for all epochs.

        Qs : list-like, optional
            optional list of values to use for the process error
            covariance; a value of None in any position will cause the filter
            to use `self.Q` for that time step. If Qs is None then self.Q is
            used for all epochs.

        Hs : list-like, optional
            optional list of values to use for the measurement matrix;
            a value of None in any position will cause the filter
            to use `self.H` for that time step. If Hs is None then self.H is
            used for all epochs.

        Rs : list-like, optional
            optional list of values to use for the measurement error
            covariance; a value of None in any position will cause the filter
            to use `self.R` for that time step. If Rs is None then self.R is
            used for all epochs.

        Bs : list-like, optional
            optional list of values to use for the control transition matrix;
            a value of None in any position will cause the filter
            to use `self.B` for that time step. If Bs is None then self.B is
            used for all epochs.

        us : list-like, optional
            optional list of values to use for the control input vector;
            a value of None in any position will cause the filter to use
            0 for that time step.

        update_first : bool, optional,
            controls whether the order of operations is update followed by
            predict, or predict followed by update. Default is predict->update.

        Returns
        -------

        means : np.array((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        means_predictions : np.array((n,dim_x,1))
            array of the state for each time step after the predictions. Each
            entry is an np.array. In other words `means[k,:]` is the state at
            step `k`.

        covariance_predictions : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the prediction.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]
            Fs = [kf.F for t in range (40)]
            Hs = [kf.H for t in range (40)]

            (mu, cov, _, _) = kf.batch_filter(zs, Rs=R_list, Fs=Fs, Hs=Hs, Qs=None,
                                              Bs=None, us=None, update_first=False)
            (xs, Ps, Ks) = kf.rts_smoother(mu, cov, Fs=Fs, Qs=None)

        """

        n = np.size(zs,0)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Rs is None:
            Rs = [self.R] * n
        if Bs is None:
            Bs = [self.B] * n
        if us is None:
            us = [0] * n

        if len(Fs) < n: Fs = [Fs]*n
        if len(Qs) < n: Qs = [Qs]*n
        if len(Hs) < n: Hs = [Hs]*n
        if len(Rs) < n: Rs = [Rs]*n
        if len(Bs) < n: Bs = [Bs]*n
        if len(us) < n: us = [us]*n


        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means   = zeros((n, self.dim_x))
            means_p = zeros((n, self.dim_x))
        else:
            means   = zeros((n, self.dim_x, 1))
            means_p = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances   = zeros((n, self.dim_x, self.dim_x))
        covariances_p = zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.update(z, R=R, H=H)
                means[i,:]         = self.x
                covariances[i,:,:] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i,:]         = self.x
                covariances_p[i,:,:] = self.P
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i,:]         = self.x
                covariances_p[i,:,:] = self.P

                self.update(z, R=R, H=H)
                means[i,:]         = self.x
                covariances[i,:,:] = self.P

        return (means, covariances, means_p, covariances_p)



    def rts_smoother(self, Xs, Ps, Fs=None, Qs=None):
        """ Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by a Kalman filter. The usual input
        would come from the output of `KalmanFilter.batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Fs : list-like collection of numpy.array, optional
            State transition matrix of the Kalman filter at each time step.
            Optional, if not provided the filter's self.F will be used

        Qs : list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        Returns
        -------

        'x' : numpy.ndarray
           smoothed means

        'P' : numpy.ndarray
           smoothed state covariances

        'K' : numpy.ndarray
            smoother gain at each step

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K) = rts_smoother(mu, cov, kf.F, kf.Q)

        """

        assert len(Xs) == len(Ps)
        shape = Xs.shape
        n = shape[0]
        dim_x = shape[1]

        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        K = zeros((n,dim_x,dim_x))

        x, P = Xs.copy(), Ps.copy()

        for k in range(n-2,-1,-1):
            P_pred = dot3(Fs[k+1], P[k], Fs[k+1].T) + Qs[k+1]

            K[k]  = dot3(P[k], Fs[k+1].T, linalg.inv(P_pred))
            x[k] += dot(K[k], x[k+1] - dot(Fs[k+1], x[k]))
            P[k] += dot3(K[k], P[k+1] - P_pred, K[k].T)

        return (x, P, K)


    def get_prediction(self, u=0):
        """ Predicts the next state of the filter and returns it. Does not
        alter the state of the filter.

        Parameters
        ----------

        u : np.array
            optional control input

        Returns
        -------

        (x, P) : tuple
            State vector and covariance array of the prediction.
        """

        x = dot(self.F, self.x) + dot(self.B, u)
        P = self._alpha_sq * dot3(self.F, self.P, self.F.T) + self.Q
        return (x, P)


    def residual_of(self, z):
        """ returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return z - dot(self.H, self.x)


    def measurement_of_state(self, x):
        """ Helper function that converts a state into a measurement.

        Parameters
        ----------

        x : np.array
            kalman state vector

        Returns
        -------

        z : np.array
            measurement corresponding to the given state
        """

        return dot(self.H, x)


    @property
    def alpha(self):
        """ Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1].

        References
        ----------

        [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
            p. 208-212. (2006)
        """

        return self._alpha_sq**.5


    @property
    def likelihood(self):
        """ likelihood of measurement"""
        return math.exp(self.log_likelihood)


    @alpha.setter
    def alpha(self, value):
        assert np.isscalar(value)
        assert value > 0

        self._alpha_sq = value**2


def update(x, P, z, R, H=None, return_all=False):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.

    This can handle either the multidimensional or unidimensional case. If
    all parameters are floats instead of arrays the filter will still work,
    and return floats for x, P as the result.

    update(1, 2, 1, 1, 1)  # univariate
    update(x, P, 1



    Parameters
    ----------

    x : numpy.array(dim_x, 1), or float
        State estimate vector

    P : numpy.array(dim_x, dim_x), or float
        Covariance matrix

    z : numpy.array(dim_z, 1), or float
        measurement for this update.

    R : numpy.array(dim_z, dim_z), or float
        Measurement noise matrix

    H : numpy.array(dim_x, dim_x), or float, optional
        Measurement function. If not provided, a value of 1 is assumed.

    return_all : bool, default False
        If true, y, K, S, and log_likelihood are returned, otherwise
        only x and P are returned.

    Returns
    -------

    x : numpy.array
        Posterior state estimate vector

    P : numpy.array
        Posterior covariance matrix

    y : numpy.array or scalar
        Residua. Difference between measurement and state in measurement space

    K : numpy.array
        Kalman gain

    S : numpy.array
        System uncertainty in measurement space

    log_likelihood : float
        log likelihood of the measurement
    """


    if z is None:
        if return_all:
            return x, P, None, None, None, None
        else:
            return x, P

    if H is None:
        H = np.array([1])

    if np.isscalar(H):
        H = np.array([H])

    if not np.isscalar(x):
        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

    # error (residual) between measurement and prediction
    y = z - dot(H, x)

    # project system uncertainty into measurement space
    S = dot3(H, P, H.T) + R


    # map system uncertainty into kalman gain
    try:
        K = dot3(P, H.T, linalg.inv(S))
    except:
        K = dot3(P, H.T, 1/S)


    # predict new x with residual scaled by the kalman gain
    x = x + dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = dot(K, H)


    try:
        I_KH = np.eye(KH.shape[0]) - KH
    except:
        I_KH = np.array(1 - KH)
    P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    if return_all:
        # compute log likelihood
        log_likelihood = logpdf(z, dot(H, x), S)
        return x, P, y, K, S, log_likelihood
    else:
        return x, P



def predict(x, P, F=1, Q=0, u=0, B=1, alpha=1.):
    """ Predict next position using the Kalman filter state propagation
    equations.

    Parameters
    ----------

    x : numpy.array
        State estimate vector

    P : numpy.array
        Covariance matrix

    F : numpy.array()
        State Transition matrix

    Q : numpy.array
        Process noise matrix


    u : numpy.array, default 0.
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.

    B : numpy.array, default 0.
        Optional control transition matrix.

    alpha : float, default=1.0
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon

    Returns
    -------

    x : numpy.array
        Prior state estimate vector

    P : numpy.array
        Prior covariance matrix
    """

    if np.isscalar(F):
        F = np.array(F)
    x = dot(F, x) + dot(B, u)
    P = (alpha * alpha) * dot3(F, P, F.T) + Q

    return x, P



def batch_filter(x, P, zs, Fs, Qs, Hs, Rs, Bs=None, us=None, update_first=False):
    """ Batch processes a sequences of measurements.

    Parameters
    ----------

    zs : list-like
        list of measurements at each time step. Missing measurements must be
        represented by 'None'.

    Fs : list-like
        list of values to use for the state transition matrix matrix;
        a value of None in any position will cause the filter
        to use `self.F` for that time step.

    Qs : list-like,
        list of values to use for the process error
        covariance; a value of None in any position will cause the filter
        to use `self.Q` for that time step.

    Hs : list-like, optional
        list of values to use for the measurement matrix;
        a value of None in any position will cause the filter
        to use `self.H` for that time step.

    Rs : list-like, optional
        list of values to use for the measurement error
        covariance; a value of None in any position will cause the filter
        to use `self.R` for that time step.

    Bs : list-like, optional
        list of values to use for the control transition matrix;
        a value of None in any position will cause the filter
        to use `self.B` for that time step.

    us : list-like, optional
        list of values to use for the control input vector;
        a value of None in any position will cause the filter to use
        0 for that time step.

    update_first : bool, optional,
        controls whether the order of operations is update followed by
        predict, or predict followed by update. Default is predict->update.


    Returns
    -------

    means : np.array((n,dim_x,1))
        array of the state for each time step after the update. Each entry
        is an np.array. In other words `means[k,:]` is the state at step
        `k`.

    covariance : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the update.
        In other words `covariance[k,:,:]` is the covariance at step `k`.

    means_predictions : np.array((n,dim_x,1))
        array of the state for each time step after the predictions. Each
        entry is an np.array. In other words `means[k,:]` is the state at
        step `k`.

    covariance_predictions : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the prediction.
        In other words `covariance[k,:,:]` is the covariance at step `k`.

    Examples
    --------

    .. code-block:: Python

        zs = [t + random.randn()*4 for t in range (40)]
        Fs = [kf.F for t in range (40)]
        Hs = [kf.H for t in range (40)]

        (mu, cov, _, _) = kf.batch_filter(zs, Rs=R_list, Fs=Fs, Hs=Hs, Qs=None,
                                          Bs=None, us=None, update_first=False)
        (xs, Ps, Ks) = kf.rts_smoother(mu, cov, Fs=Fs, Qs=None)

    """

    n = np.size(zs,0)
    dim_x = x.shape[0]

    # mean estimates from Kalman Filter
    if x.ndim == 1:
        means   = zeros((n, dim_x))
        means_p = zeros((n, dim_x))
    else:
        means   = zeros((n, dim_x, 1))
        means_p = zeros((n, dim_x, 1))

    # state covariances from Kalman Filter
    covariances   = zeros((n, dim_x, dim_x))
    covariances_p = zeros((n, dim_x, dim_x))

    if us is None:
       us = [0.]*n
       Bs = [0.]*n

    if len(Fs) < n: Fs = [Fs]*n
    if len(Qs) < n: Qs = [Qs]*n
    if len(Hs) < n: Hs = [Hs]*n
    if len(Rs) < n: Rs = [Rs]*n
    if len(Bs) < n: Bs = [Bs]*n
    if len(us) < n: us = [us]*n


    if update_first:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = update(x, P, z, R=R, H=H)
            means[i,:]         = x
            covariances[i,:,:] = P

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i,:]         = x
            covariances_p[i,:,:] = P
    else:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i,:]         = x
            covariances_p[i,:,:] = P

            x, P  = update(x, P, z, R=R, H=H)
            means[i,:]         = x
            covariances[i,:,:] = P

    return (means, covariances, means_p, covariances_p)



def rts_smoother(Xs, Ps, Fs, Qs):
    """ Runs the Rauch-Tung-Striebal Kalman smoother on a set of
    means and covariances computed by a Kalman filter. The usual input
    would come from the output of `KalmanFilter.batch_filter()`.

    Parameters
    ----------

    Xs : numpy.array
       array of the means (state variable x) of the output of a Kalman
       filter.

    Ps : numpy.array
        array of the covariances of the output of a kalman filter.

    Fs : list-like collection of numpy.array
        State transition matrix of the Kalman filter at each time step.
        Optional, if not provided the filter's self.F will be used

    Qs : list-like collection of numpy.array, optional
        Process noise of the Kalman filter at each time step. Optional,
        if not provided the filter's self.Q will be used

    Returns
    -------

    'x' : numpy.ndarray
       smoothed means

    'P' : numpy.ndarray
       smoothed state covariances

    'K' : numpy.ndarray
        smoother gain at each step


    Examples
    --------

    .. code-block:: Python

        zs = [t + random.randn()*4 for t in range (40)]

        (mu, cov, _, _) = kalman.batch_filter(zs)
        (x, P, K) = rts_smoother(mu, cov, kf.F, kf.Q)
    """

    assert len(Xs) == len(Ps)
    n = Xs.shape[0]
    dim_x = Xs.shape[1]

    # smoother gain
    K = zeros((n,dim_x,dim_x))
    x, P = Xs.copy(), Ps.copy()

    for k in range(n-2,-1,-1):
        P_pred = dot3(Fs[k], P[k], Fs[k].T) + Qs[k]

        K[k]  = dot3(P[k], Fs[k].T, linalg.inv(P_pred))
        x[k] += dot (K[k], x[k+1] - dot(Fs[k], x[k]))
        P[k] += dot3 (K[k], P[k+1] - P_pred, K[k].T)

    return (x, P, K)
