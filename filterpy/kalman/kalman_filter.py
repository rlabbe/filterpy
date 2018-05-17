# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments, too-many-branches,
# pylint: disable=too-many-locals, too-many-instance-attributes, too-many-lines

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
prediction is called the *prior*, which you can think of colloquially
as the estimate prior to incorporating the measurement.

The update step, implemented with the method or function `update()`,
incorporates the measurement z with covariance R, into the state
estimate (x, P). The class stores the system uncertainty in S,
the innovation (residual between prediction and measurement in
measurement space) in y, and the Kalman gain in k. The procedural
form returns these variables to you. In Bayesian terms this computes
the *posterior* - the estimate after the information from the
measurement is incorporated.

Whether you use the OO form or procedural form is up to you. If
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

import sys
import warnings
import math
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z, repeated_array



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

    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        State estimate

    P : numpy.array(dim_x, dim_x)
        State covariance matrix

    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix

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
        System uncertainty projected to measurement space. Read only.

    z : ndarray
        Last measurement used in update(). Read only.

    log_likelihood : float
        log-likelihood of the last measurement. Read only.

    likelihood : float
        likelihood of last measurement. Read only.

        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.

    mahalanobis : float
        mahalanobis distance of the innovation. Read only.

    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv

        This is only used to invert self.S. If you know it is diagonal, you
        might choose to set it to filterpy.common.inv_diagonal, which is
        several times faster than numpy.linalg.inv for diagonal matrices.

    alpha : float 
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.

        References
        ----------

        .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
           p. 208-212. (2006)

    
    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))        # state
        self.P = eye(dim_x)               # uncertainty covariance
        self.Q = eye(dim_x)               # process uncertainty
        self.B = None                     # control transition matrix
        self.F = eye(dim_x)               # state transition matrix
        self.H = zeros((dim_z, dim_x))    # Measurement function
        self.R = eye(dim_z)               # state uncertainty
        self._alpha_sq = 1.               # fading memory control
        self.M = np.zeros((dim_z, dim_z)) # process-measurement cross correlation

        self.z = reshape_z(zeros((dim_z)), self.dim_z, self.x.ndim)

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = math.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.inv = np.linalg.inv


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
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
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

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()


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

        z = reshape_z(z, self.dim_z, self.x.ndim)

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
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        self.z = z.copy() # save the measurement

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None


    @property
    def log_likelihood(self):
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood


    @property
    def likelihood(self):
        if self._likelihood is None:
            self._likelihood = math.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood


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
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        """

        if B is None:
            B = self.B

        # x = Fx + Bu
        if B is not None:
            self.x = dot(self.F, self.x) + dot(B, u)
        else:
            self.x = dot(self.F, self.x)

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()


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
        >>> saved_k = np.copy(cv.K)
        >>> saved_P = np.copy(cv.P)

        later on:

        >>> cv = kinematic_kf(dim=3, order=2) # 3D const velocity filter
        >>> cv.K = np.copy(saved_K)
        >>> cv.P = np.copy(saved_P)
        >>> for i in range(100):
        >>>     cv.predict_steadystate()
        >>>     cv.update_steadystate([i, i, i])
        """

        if z is None:
            return

        z = reshape_z(z, self.dim_z, self.x.ndim)

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(self.H, self.x)


        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        self.z = z.copy() # save the measurement

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None


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

        z = reshape_z(z, self.dim_z, self.x.ndim)

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
        self.SI = self.inv(self.S)

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT + self.M, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(H, self.P) + self.M.T)

        self.z = z.copy() # save the measurement
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None


    def batch_filter(self, zs, Fs=None, Qs=None, Hs=None,
                     Rs=None, Bs=None, us=None, update_first=False,
                     saver=None):
        """ Batch processes a sequences of measurements.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self.dt`. Missing
            measurements must be represented by `None`.

        Fs : None, np.array or list-like, default=None
            optional value or list of values to use for the state transition
            matrix F.

            If Fs is None then self.F is used for all epochs.

            If Fs contains a single matrix, then it is used as F for all
            epochs.

            If it is a list of matrices or a 3D array where
            len(Fs) == len(zs), then it is treated as a list of F values, one
            per epoch. This allows you to have varying F per epoch.

        Qs : None, np.array or list-like, default=None
            optional value or list of values to use for the process error
            covariance Q.

            If Qs is None then self.Q is used for all epochs.

            If Qs contains a single matrix, then it is used as Q for all
            epochs.

            If it is a list of matrices or a 3D array where
            len(Qs) == len(zs), then it is treated as a list of Q values, one
            per epoch. This allows you to have varying Q per epoch.


        Hs : None, np.array or list-like, default=None
            optional list of values to use for the measurement matrix H.

            If Hs is None then self.H is used for all epochs.

            If Hs contains a single matrix, then it is used as H for all
            epochs.

            If it is a list of matrices or a 3D array where
            len(Hs) == len(zs), then it is treated as a list of H values, one
            per epoch. This allows you to have varying H per epoch.


        Rs : None, np.array or list-like, default=None
            optional list of values to use for the measurement error
            covariance R.

            If Rs is None then self.R is used for all epochs.

            If Rs contains a single matrix, then it is used as H for all
            epochs.

            If it is a list of matrices or a 3D array where
            len(Rs) == len(zs), then it is treated as a list of R values, one
            per epoch. This allows you to have varying R per epoch.


        Bs : None, np.array or list-like, default=None
            optional list of values to use for the control transition matrix B.

            If Bs is None then self.B is used for all epochs.

            If Bs contains a single matrix, then it is used as B for all
            epochs.

            If it is a list of matrices or a 3D array where
            len(Bs) == len(zs), then it is treated as a list of B values, one
            per epoch. This allows you to have varying B per epoch.


        us : None, np.array or list-like, default=None
            optional list of values to use for the control input vector;

            If us is None then None is used for all epochs (equivalent to 0,
            or no control input).

            If us contains a single matrix, then it is used as H for all
            epochs.

            If it is a list of matrices or a 3D array where
            len(Rs) == len(zs), then it is treated as a list of R values, one
            per epoch. This allows you to have varying R per epoch.


        update_first : bool, optional, default=False
            controls whether the order of operations is update followed by
            predict, or predict followed by update. Default is predict->update.

        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch

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

            # this example demonstrates tracking a measurement where the time
            # between measurement varies, as stored in dts. This requires
            # that F be recomputed for each epoch. The output is then smoothed
            # with an RTS smoother.

            zs = [t + random.randn()*4 for t in range (40)]
            Fs = [np.array([[1., dt], [0, 1]] for dt in dts]

            (mu, cov, _, _) = kf.batch_filter(zs, Fs=Fs)
            (xs, Ps, Ks) = kf.rts_smoother(mu, cov, Fs=Fs)
        """

        #pylint: disable=too-many-statements
        n = np.size(zs, 0)
        if Fs is None:
            Fs = self.F
        if Qs is None:
            Qs = self.Q
        if Hs is None:
            Hs = self.H
        if Rs is None:
            Rs = self.R
        if Bs is None:
            Bs = self.B
        if us is None:
            us = 0

        Fs = repeated_array(Fs, n)
        Qs = repeated_array(Qs, n)
        Hs = repeated_array(Hs, n)
        Rs = repeated_array(Rs, n)
        Bs = repeated_array(Bs, n)
        us = repeated_array(us, n)


        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((n, self.dim_x))
            means_p = zeros((n, self.dim_x))
        else:
            means = zeros((n, self.dim_x, 1))
            means_p = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((n, self.dim_x, self.dim_x))
        covariances_p = zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                if saver is not None:
                    saver.save()
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                if saver is not None:
                    saver.save()

        return (means, covariances, means_p, covariances_p)


    def rts_smoother(self, Xs, Ps, Fs=None, Qs=None, inv=np.linalg.inv):
        """
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
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

        inv : function, default numpy.linalg.inv
            If you prefer another inverse function, such as the Moore-Penrose
            pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv


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

        if len(Xs) != len(Ps):
            raise ValueError('length of Xs and Ps must be the same')

        n = Xs.shape[0]
        dim_x = Xs.shape[1]

        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        K = zeros((n, dim_x, dim_x))

        x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n-2, -1, -1):
            Pp[k] = dot(dot(Fs[k+1], P[k]), Fs[k+1].T) + Qs[k+1]

            #pylint: disable=bad-whitespace
            K[k]  = dot(dot(P[k], Fs[k+1].T), inv(Pp[k]))
            x[k] += dot(K[k], x[k+1] - dot(Fs[k+1], x[k]))
            P[k] += dot(dot(K[k], P[k+1] - Pp[k]), K[k].T)

        return (x, P, K, Pp)


    def get_prediction(self, u=0):
        """
        Predicts the next state of the filter and returns it without
        altering the state of the filter.

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
        """
        Computes the new estimate based on measurement `z` and returns it
        without altering the state of the filter.

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
        z = reshape_z(z, self.dim_z, self.x.ndim)

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
        K = dot(PHT, self.inv(S))

        # predict new x with residual scaled by the kalman gain
        x = x + dot(K, y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(K, H)
        P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

        return x, P


    def residual_of(self, z):
        """
        Returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return z - dot(self.H, self.x_prior)


    def measurement_of_state(self, x):
        """
        Helper function that converts a state into a measurement.

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
    def mahalanobis(self):
        """"
        Mahalanobis distance of innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = float(np.dot(np.dot(self.y.T, self.SI), self.y))
        return self._mahalanobis


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


    def log_likelihood_of(self, z):
        """
        log likelihood of the measurement `z`. This should only be called
        after a call to update(). Calling after predict() will yield an
        incorrect result."""

        if z is None:
            return math.log(sys.float_info.min)
        return logpdf(z, dot(self.H, self.x), self.S)


    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value) or value < 1:
            raise ValueError('alpha must be a float greater than 1')

        self._alpha_sq = value**2


    def __repr__(self):
        return '\n'.join([
            'KalmanFilter object',
            pretty_str('dim_x', self.dim_x),
            pretty_str('dim_z', self.dim_z),
            pretty_str('dim_u', self.dim_u),
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('x_post', self.x_post),
            pretty_str('P_post', self.P_post),
            pretty_str('F', self.F),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('H', self.H),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('S', self.S),
            pretty_str('SI', self.SI),
            pretty_str('M', self.M),
            pretty_str('B', self.B),
            pretty_str('z', self.z),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('alpha', self.alpha),
            pretty_str('inv', self.inv)
            ])


    def test_matrix_dimensions(self, z=None, H=None, R=None, F=None, Q=None):
        """
        Performs a series of asserts to check that the size of everything
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

        if H is None:
            H = self.H
        if R is None:
            R = self.R
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        x = self.x
        P = self.P

        assert x.ndim == 1 or x.ndim == 2, \
                "x must have one or two dimensions, but has {}".format(x.ndim)

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
            assert r_shape == () or r_shape == (1,) or r_shape == (1, 1), \
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
            assert Hx.ndim == 1 or shape(Hx) == (1, 1), \
            "shape of z should be {}, not {} for the given H".format(
                shape(Hx), z_shape)

        elif shape(Hx) == (1,):
            assert z_shape[0] == 1, 'Shape of z must be {} for the given H'.format(shape(Hx))

        else:
            assert (z_shape == shape(Hx) or
                    (len(z_shape) == 1 and shape(Hx) == (z_shape[0], 1))), \
                    "shape of z should be {}, not {} for the given H".format(
                        shape(Hx), z_shape)

        if np.ndim(Hx) > 1 and shape(Hx) != (1, 1):
            assert shape(Hx) == z_shape, \
               'shape of z should be {} for the given H, but it is {}'.format(
                   shape(Hx), z_shape)


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

    #pylint: disable=bare-except

    if z is None:
        if return_all:
            return x, P, None, None, None, None
        return x, P

    if H is None:
        H = np.array([1])

    if np.isscalar(H):
        H = np.array([H])

    Hx = np.atleast_1d(dot(H, x))
    z = reshape_z(z, Hx.shape[0], x.ndim)

    # error (residual) between measurement and prediction
    y = z - Hx

    # project system uncertainty into measurement space
    S = dot(dot(H, P), H.T) + R


    # map system uncertainty into kalman gain
    try:
        K = dot(dot(P, H.T), linalg.inv(S))
    except:
        # can't invert a 1D array, annoyingly
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
    z = reshape_z(z, Hx.shape[0], x.ndim)

    # error (residual) between measurement and prediction
    y = z - Hx

    # estimate new x with residual scaled by the kalman gain
    return x + dot(K, y)


def predict(x, P, F=1, Q=0, u=0, B=1, alpha=1.):
    """
    Predict next state (prior) using the Kalman filter state propagation
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
    """
    Predict next state (prior) using the Kalman filter state propagation
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



def batch_filter(x, P, zs, Fs, Qs, Hs, Rs, Bs=None, us=None,
                 update_first=False, saver=None):
    """
    Batch processes a sequences of measurements.

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

        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch

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

    n = np.size(zs, 0)
    dim_x = x.shape[0]

    # mean estimates from Kalman Filter
    if x.ndim == 1:
        means = zeros((n, dim_x))
        means_p = zeros((n, dim_x))
    else:
        means = zeros((n, dim_x, 1))
        means_p = zeros((n, dim_x, 1))

    # state covariances from Kalman Filter
    covariances = zeros((n, dim_x, dim_x))
    covariances_p = zeros((n, dim_x, dim_x))

    if us is None:
        us = [0.] * n
        Bs = [0.] * n

    #pylint: disable=multiple-statements
    Fs = repeated_array(Fs, n)
    Qs = repeated_array(Qs, n)
    Hs = repeated_array(Hs, n)
    Rs = repeated_array(Rs, n)
    Bs = repeated_array(Bs, n)
    us = repeated_array(us, n)


    if update_first:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
            if saver is not None:
                saver.save()
    else:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P

            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P
            if saver is not None:
                saver.save()

    return (means, covariances, means_p, covariances_p)



def rts_smoother(Xs, Ps, Fs, Qs):
    """
    Runs the Rauch-Tung-Striebal Kalman smoother on a set of
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

    if len(Xs) != len(Ps):
        raise ValueError('length of Xs and Ps must be the same')

    n = Xs.shape[0]
    dim_x = Xs.shape[1]

    # smoother gain
    K = zeros((n, dim_x, dim_x))
    x, P, pP = Xs.copy(), Ps.copy(), Ps.copy()

    for k in range(n-2, -1, -1):
        pP[k] = dot(dot(Fs[k], P[k]), Fs[k].T) + Qs[k]

        #pylint: disable=bad-whitespace
        K[k]  = dot(dot(P[k], Fs[k].T), linalg.inv(pP[k]))
        x[k] += dot(K[k], x[k+1] - dot(Fs[k], x[k]))
        P[k] += dot(dot(K[k], P[k+1] - pP[k]), K[k].T)

    return (x, P, K, pP)


class Saver(object):
    """
    Deprecated. Use filterpy.common.Saver instead.

    Helper class to save the states of the KalmanFilter class.
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

        warnings.warn(
            'Use filterpy.common.Saver instead of this, as it works for any filter clase',
            DeprecationWarning)

        self.xs = []
        self.Ps = []
        self.Ks = []
        self.ys = []
        self.xs_prior = []
        self.Ps_prior = []
        self.kf = kf
        if save_current:
            self.save()


    def save(self):
        """ save the current state of the Kalman filter"""

        kf = self.kf
        self.xs.append(np.copy(kf.x))
        self.Ps.append(np.copy(kf.P))
        self.Ks.append(np.copy(kf.K))
        self.ys.append(np.copy(kf.y))
        self.xs_prior.append(np.copy(kf.x_prior))
        self.Ps_prior.append(np.copy(kf.P_prior))


    def to_array(self):
        """ convert all of the lists into np.array"""

        self.xs = np.array(self.xs)
        self.Ps = np.array(self.Ps)
        self.Ks = np.array(self.Ks)
        self.ys = np.array(self.ys)
        self.xs_prior = np.array(self.xs_prior)
        self.Ps_prior = np.array(self.Ps_prior)
