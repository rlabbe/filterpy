# -*- coding: utf-8 -*-

"""
This module implements the linear Kalman filter in both an object
oriented and procedural form. The KalmanFilter class implements
the filter by storing the various matrices in instance variables,
minimizing the amount of bookkeeping you have to do.

All Kalman filters operate with a predict->update cycle. The
predict step, implemented with the method or function predict(),
uses the state transition matrix F to predict the state in the next
time period (epoch). The state is stored as a gaussian (x, P), where
x is the state (column) vector, and P is its covariance. Covariance
matrix Q specifies the process covariance. In Bayesian terms, this
prediction is called the *prior*, which you can think of collequally
as the estimate prior to incorporating the measurement.

The update step, implemented with the method or function `update()`,
incoporates the measurement z with covariance R, into the state
estimate (x, P). The class stores the system uncertainty in S,
the innovation (residual between prediction and measurement in
measurement space) in y, and the Kalman gain in k. The procedural
form returns these variables to you. In Bayesian terms this computes
the *posterior* - the estimate after the information from the
measurement is incorporated.

Whether you use the OO form or procedureal form is up to you. If
matrices such as H, R, and F are changing each epoch, you'll probably
opt to use the procedural form. If they are unchanging, the OO
form is perhaps easier to use since you won't need to keep track
of these matrices. This is especially useful if you are implementing
banks of filters or comparing various KF designs for performance;
a trivial coding bug could lead to using the wrong sets of matrices.

This module also offers an implementation of the RTS smoother, and
other helper functions, such as log likelihood computations.

The Saver class allows you to easily save the state of the
KalmanFilter class after every update

This module expects NumPy arrays for all values that expect
arrays, although in a few cases, particularly method parameters,
it will accept types that convert to NumPy arrays, such as lists
of lists. These exceptions are documented in the method or function.

Examples
--------
The following example constructs a constant velocity kinematic
filter, filters noisy data, and plots the results

.. code-block:: Python

    import matplotlib.pyplot as plt
    import numpy as np
    from filterpy.kalman import KalmanFilter, Saver
    from filterpy.common import Q_discrete_white_noise

    r_std, q_std = 2., 0.003
    cv = KalmanFilter(dim_x=2, dim_z=1)
    cv.x = np.array([[0., 1.]]) # position, velocity
    cv.F = np.array([[1, dt],[ [0, 1]])
    cv.R = np.array([[r_std^^2]])
    f.H = np.array([[1., 0.]])
    f.P = np.diag([.1^^2, .03^^2)
    f.Q = Q_discrete_white_noise(2, dt, q_std**2)

    saver = Saver(cv)
    for z in range(100):
        cv.predict()
        cv.update[[z + randn() * r_std])
        saver.save() # save the filter's state

    saver.to_array()
    plt.plot(saver.xs[:, 0])


This code implements the same filter using the procedural form

    x = np.array([[0., 1.]]) # position, velocity
    F = np.array([[1, dt],[ [0, 1]])
    R = np.array([[r_std^^2]])
    H = np.array([[1., 0.]])
    P = np.diag([.1^^2, .03^^2)
    Q = Q_discrete_white_noise(2, dt, q_std**2)

    for z in range(100):
        x, P = predict(x, P, F=F, Q=Q)
        x, P = update[x, P, z=[z + randn() * r_std], R=R, H=H)
        xs.append(x[0, 0]
    plt.plot(xs)


For more examples see the test subdirectory, or refer to the
book cited below. In it I both teach Kalman filtering from basic
principles, and teach the use of this library in great detail.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2014-2018 Roger R Labbe Jr.
"""

from __future__ import absolute_import, division
import math
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import scipy.linalg as linalg
import sys
from filterpy.stats import logpdf


class KalmanFilter(object):
    r""" Implements a Kalman filter. You are responsible for setting the
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

    H : numpy.array(dim_z, dim_x)
        Measurement function


    y : numpy.array
        Residual of the update step. Read only.

    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.

    S :  numpy.array
        Systen uncertaintly projected to measurement space. Read only.

    log_likelihood : float
        log-likelihood of the last measurement. Read only.

    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """



    def __init__(self, dim_x, dim_z, dim_u=0, compute_log_likelihood=True):
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

        compute_log_likelihood : bool (default = True)
            Computes log likelihood by default, but this can be a slow
            computation, so if you never use it you can turn this computation
            off.
        """

        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1)) # state
        self.P = eye(dim_x)        # uncertainty covariance
        self.Q = eye(dim_x)        # process uncertainty
        self.B = None              # control transition matrix
        self.F = eye(dim_x)        # state transition matrix
        self.H = zeros((dim_z, dim_x)) # Measurement function
        self.R = eye(dim_z)        # state uncertainty
        self._alpha_sq = 1.        # fading memory control
        self.M = 0.                # process-measurement cross correlation

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros(self.x.shape) # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_pred = zeros((dim_x, 1))
        self.P_pred = eye(dim_x)

        self.compute_log_likelihood = compute_log_likelihood
        self.log_likelihood = math.log(sys.float_info.min)


    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        if z is None:
            return

        z = _reshape_z(z, self.dim_z, self.x.ndim)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, linalg.inv(self.S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        if self.x.ndim == 2:
            assert self.x.shape[0] == self.dim_x and self.x.shape[1] == 1

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        if self.compute_log_likelihood:
            self.log_likelihood = logpdf(x=self.y, cov=self.S)


    def update_steadystate(self, z):
        """
        Add a new measurement (z) to the Kalman filter without recomputing
        the Kalman gain K, the state covariance P, or the system
        uncertainty S.

        You can use this for LTI systems since the Kalman gain and covariance
        converge to a fixed value. Precompute these and assign them explicitly,
        or run the Kalman filter using the normal predict()/update(0 cycle
        until they converge.

        The main advantage of this call is speed. We do significantly less
        computation, notably avoiding a costly matrix inversion.

        Use in conjunction with predict_steadystate(), otherwise P will grow
        without bound.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.


        Examples
        --------
        >>> cv = kinematic_kf(dim=3, order=2) # 3D const velocity filter
        >>> # let filter converge on representative data, then save k and P
        >>> for i in range(100):
        >>>     cv.predict()
        >>>     cv.update([i, i, i])
        >>> saved_k = cv.K[:]
        >>> saved_P = cv.P[:]

        later on:

        >>> cv = kinematic_kf(dim=3, order=2) # 3D const velocity filter
        >>> cv.K = saved_K[:]
        >>> cv.P = saved_P[:]
        >>> for i in range(100):
        >>>     cv.predict_steadystate()
        >>>     cv.update_steadystate([i, i, i])
        """

        if z is None:
            return

        z = _reshape_z(z, self.dim_z, self.x.ndim)

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(self.H, self.x)

        if self.compute_log_likelihood:
            S = dot(dot(self.H, self.P), self.H.T) + self.R

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        if self.compute_log_likelihood:
            self.log_likelihood = logpdf(x=self.y, cov=S)


    def update_correlated(self, z, R=None, H=None):
        """ Add a new measurement (z) to the Kalman filter assuming that
        process noise and measurement noise are correlated as defined in
        the `self.M` matrix.

        If z is None, nothing is changed.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array,  or None
            Optionally provide H to override the measurement function for this
            one call, otherwise  self.H will be used.
        """

        if z is None:
            return

        z = _reshape_z(z, self.dim_z)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if self.x.ndim == 1 and shape(z) == (1, 1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + dot(H, self.M) + dot(self.M.T, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT + self.M, linalg.inv(self.S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(H, self.P) + self.M.T)

        if self.compute_log_likelihood:
            self.log_likelihood = logpdf(x=self.y, cov=self.S)


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
        """
        Predict next state (prior) using the Kalman filter state propagation
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
        if B is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        self.x_pred = self.x[:]
        self.P_pred = self.P[:]


    def predict_steadystate(self, u=0, B=None):
        """
        Predict state (prior) using the Kalman filter state propagation
        equations. Only x is updated, P is left unchanged. See
        update_steadstate() for a longer explanation of when to use this
        method.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None in
            any position will cause the filter to use `self.B`.
        """

        if B is None:
            B = self.B

        # x = Fx + Bu
        if B is not None:
            self.x = dot(self.F, self.x) + dot(B, u)
        else:
            self.x = dot(self.F, self.x)
        self.x_pred = self.x[:]
        # strictly speaking not necessary, but if the user initialized k and
        # P manually, ensure it is properly set
        self.P_pred = self.P[:]


    def batch_filter(self, zs, Fs=None, Qs=None, Hs=None,
                     Rs=None, Bs=None, us=None, update_first=False):
        """ Batch processes a sequences of measurements.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self.dt`. Missing
            measurements must be represented by `None`.

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
                means[i, :]          = self.x
                covariances[i, :, :] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :]          = self.x
                covariances_p[i, :, :] = self.P
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :]          = self.x
                covariances_p[i, :, :] = self.P

                self.update(z, R=R, H=H)
                means[i, :]          = self.x
                covariances[i, :, :] = self.P

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

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Pp : numpy.ndarray
           Predicted state covariances

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K, Pp) = rts_smoother(mu, cov, kf.F, kf.Q)

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
        K = zeros((n,dim_x, dim_x))

        x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()

        for k in range(n-2,-1,-1):
            Pp[k] = dot(dot(Fs[k+1], P[k]), Fs[k+1].T) + Qs[k+1]

            K[k]  = dot(dot(P[k], Fs[k+1].T), linalg.inv(Pp[k]))
            x[k] += dot(K[k], x[k+1] - dot(Fs[k+1], x[k]))
            P[k] += dot(dot(K[k], P[k+1] - Pp[k]), K[k].T)

        return (x, P, K, Pp)


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
        P = self._alpha_sq * dot(dot(self.F, self.P), self.F.T) + self.Q
        return (x, P)


    def get_update(self, z=None):
        """ Computes the new estimate based on measurement `z`. Does not
        alter the state of the filter.

        Parameters
        ----------

        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        Returns
        -------

        (x, P) : tuple
            State vector and covariance array of the update.

       """

        if z is None:
            return self.x, self.P
        z = _reshape_z(z, self.dim_z)

        R = self.R
        H = self.H
        P = self.P
        x = self.x

        # error (residual) between measurement and prediction
        y = z - dot(H, x)

        # common subexpression for speed
        PHT = dot(P, H.T)

        # project system uncertainty into measurement space
        S = dot(H, PHT) + R

        # map system uncertainty into kalman gain
        K = dot(PHT, linalg.inv(S))

        # predict new x with residual scaled by the kalman gain
        x = x + dot(K, y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - dot(K, H)
        P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

        return x, P


    def residual_of(self, z):
        """ returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return z - dot(self.H, self.x_pred)


    def measurement_of_state(self, x):
        """ Helper function that converts a state into a measurement.

        Parameters
        ----------

        x : np.array
            kalman state vector

        Returns
        -------

        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        """

        return dot(self.H, x)


    @property
    def alpha(self):
        """
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.

        References
        ----------

        .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
           p. 208-212. (2006)

        """

        return self._alpha_sq**.5


    @property
    def likelihood(self):
        """
        likelihood of last measurment.

        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.

        But really, this is a bad measure because of the scaling that is
        involved - try to use log-likelihood in your equations!"""

        lh = math.exp(self.log_likelihood)
        if lh == 0:
             lh = sys.float_info.min
        return lh


    def log_likelihood_of(self, z):
        """
        log likelihood of the measurement `z`. This should only be called
        after a call to update(). Calling after predict() will yield an
        incorrect result."""

        if z is None:
            return math.log(sys.float_info.min)
        else:
            return logpdf(z, dot(self.H, self.x), self.S)


    @alpha.setter
    def alpha(self, value):
        assert np.isscalar(value)
        assert value > 0.

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

    z : (dim_z, 1): array_like
        measurement for this update. z can be a scalar if dim_z is 1,
        otherwise it must be convertible to a column vector.

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

    Hx = np.atleast_1d(dot(H, x))
    z = _reshape_z(z, Hx.shape[0], x.ndim)

    # error (residual) between measurement and prediction
    y = z - Hx

    # project system uncertainty into measurement space
    S = dot(dot(H, P), H.T) + R


    # map system uncertainty into kalman gain
    try:
        K = dot(dot(P, H.T), linalg.inv(S))
    except:
        K = dot(dot(P, H.T), 1./S)


    # predict new x with residual scaled by the kalman gain
    x = x + dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = dot(K, H)


    try:
        I_KH = np.eye(KH.shape[0]) - KH
    except:
        I_KH = np.array([1 - KH])
    P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)


    if return_all:
        # compute log likelihood
        log_likelihood = logpdf(z, dot(H, x), S)
        return x, P, y, K, S, log_likelihood
    else:
        return x, P


def update_steadystate(x, z, K, H=None):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.


    Parameters
    ----------

    x : numpy.array(dim_x, 1), or float
        State estimate vector


    z : (dim_z, 1): array_like
        measurement for this update. z can be a scalar if dim_z is 1,
        otherwise it must be convertible to a column vector.

    K : numpy.array, or float
        Kalman gain matrix

    H : numpy.array(dim_x, dim_x), or float, optional
        Measurement function. If not provided, a value of 1 is assumed.

    Returns
    -------

    x : numpy.array
        Posterior state estimate vector

    Examples
    --------

    This can handle either the multidimensional or unidimensional case. If
    all parameters are floats instead of arrays the filter will still work,
    and return floats for x, P as the result.

    >>> update_steadystate(1, 2, 1)  # univariate
    >>> update_steadystate(x, P, z, H)
    """


    if z is None:
        return x

    if H is None:
        H = np.array([1])

    if np.isscalar(H):
        H = np.array([H])

    Hx = np.atleast_1d(dot(H, x))
    z = _reshape_z(z, Hx.shape[0], x.ndim)

    # error (residual) between measurement and prediction
    y = z - Hx

    # estimate new x with residual scaled by the kalman gain
    return x + dot(K, y)


def predict(x, P, F=1, Q=0, u=0, B=1, alpha=1.):
    """ Predict next state (prior) using the Kalman filter state propagation
    equations.

    Parameters
    ----------

    x : numpy.array
        State estimate vector

    P : numpy.array
        Covariance matrix

    F : numpy.array()
        State Transition matrix

    Q : numpy.array, Optional
        Process noise matrix


    u : numpy.array, Optional, default 0.
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.

    B : numpy.array, optional, default 0.
        Control transition matrix.

    alpha : float, Optional, default=1.0
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
    P = (alpha * alpha) * dot(dot(F, P), F.T) + Q

    return x, P


def predict_steadystate(x, F=1, u=0, B=1):
    """ Predict next state (prior) using the Kalman filter state propagation
    equations. This steady state form only computes x, assuming that the
    covariance is constant.

    Parameters
    ----------

    x : numpy.array
        State estimate vector

    P : numpy.array
        Covariance matrix

    F : numpy.array()
        State Transition matrix

    u : numpy.array, Optional, default 0.
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.

    B : numpy.array, optional, default 0.
        Control transition matrix.

    Returns
    -------

    x : numpy.array
        Prior state estimate vector
    """

    if np.isscalar(F):
        F = np.array(F)
    x = dot(F, x) + dot(B, u)

    return x



def batch_filter(x, P, zs, Fs, Qs, Hs, Rs, Bs=None, us=None, update_first=False):
    """ Batch processes a sequences of measurements.

    Parameters
    ----------

    zs : list-like
        list of measurements at each time step. Missing measurements must be
        represented by None.

    Fs : list-like
        list of values to use for the state transition matrix matrix.

    Qs : list-like
        list of values to use for the process error
        covariance.

    Hs : list-like
        list of values to use for the measurement matrix.

    Rs : list-like
        list of values to use for the measurement error
        covariance.

    Bs : list-like, optional
        list of values to use for the control transition matrix;
        a value of None in any position will cause the filter
        to use `self.B` for that time step.

    us : list-like, optional
        list of values to use for the control input vector;
        a value of None in any position will cause the filter to use
        0 for that time step.

    update_first : bool, optional
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
            means[i, :]          = x
            covariances[i, :, :] = P

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :]          = x
            covariances_p[i, :, :] = P
    else:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :]          = x
            covariances_p[i, :, :] = P

            x, P  = update(x, P, z, R=R, H=H)
            means[i, :]          = x
            covariances[i, :, :] = P

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

    Qs : list-like collection of numpy.array, optional
        Process noise of the Kalman filter at each time step.

    Returns
    -------

    x : numpy.ndarray
       smoothed means

    P : numpy.ndarray
       smoothed state covariances

    K : numpy.ndarray
        smoother gain at each step

    pP : numpy.ndarray
       predicted state covariances

    Examples
    --------

    .. code-block:: Python

        zs = [t + random.randn()*4 for t in range (40)]

        (mu, cov, _, _) = kalman.batch_filter(zs)
        (x, P, K, pP) = rts_smoother(mu, cov, kf.F, kf.Q)
    """

    assert len(Xs) == len(Ps)
    n = Xs.shape[0]
    dim_x = Xs.shape[1]

    # smoother gain
    K = zeros((n,dim_x,dim_x))
    x, P, pP = Xs.copy(), Ps.copy(), Ps.copy()

    for k in range(n-2, -1, -1):
        pP[k] = dot(dot(Fs[k], P[k]), Fs[k].T) + Qs[k]

        K[k]  = dot(dot(P[k], Fs[k].T), linalg.inv(pP[k]))
        x[k] += dot(K[k], x[k+1] - dot(Fs[k], x[k]))
        P[k] += dot(dot(K[k], P[k+1] - pP[k]), K[k].T)

    return (x, P, K, pP)


class Saver(object):
    """ Helper class to save the states of the KalmanFilter class.
    Each time you call save() the current states are appended to lists.
    Generally you would do this once per epoch - predict/update.

    Once you are done filtering you can optionally call to_array()
    to convert all of the lists to numpy arrays. You cannot safely call
    save() after calling to_array().

    Examples
    --------

    .. code-block:: Python

        kf = KalmanFilter(...whatever)
        # initialize kf here

        saver = Saver(kf) # save data for kf filter
        for z in zs:
            kf.predict()
            kf.update(z)

            saver.save()

        saver.to_array()
        # plot the 0th element of the state
        plt.plot(saver.xs[:, 0, 0])
    """

    def __init__(self, kf, save_current=True):
        """ Construct the save object, optionally saving the current
        state of the filter"""

        self.xs = []
        self.Ps = []
        self.Ks = []
        self.ys = []
        self.xs_pred = []
        self.Ps_pred = []
        self.kf = kf
        if save_current:
            self.save()

    def save(self):
        """ save the current state of the Kalman filter"""

        kf = self.kf
        self.xs.append(kf.x[:])
        self.Ps.append(kf.P[:])
        self.Ks.append(kf.K[:])
        self.ys.append(kf.y[:])
        self.xs_pred.append(kf.x_pred[:])
        self.Ps_pred.append(kf.P_pred[:])


    def to_array(self):
        """ convert all of the lists into np.array"""

        self.xs = np.array(self.xs)
        self.Ps = np.array(self.Ps)
        self.Ks = np.array(self.Ks)
        self.ys = np.array(self.ys)
        self.xs_pred = np.array(self.xs_pred)
        self.Ps_pred = np.array(self.Ps_pred)


def _reshape_z(z, dim_z, ndim):
    """ ensure z is a (dim_z, 1) shaped vector"""

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

    if ndim == 1:
        z = z[:,0]

    if ndim == 0:
        z = z[0,0]

    return z

