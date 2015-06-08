# -*- coding: utf-8 -*-
"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
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

from filterpy.common import dot3
from filterpy.kalman import unscented_transform
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import inv, cholesky


class UnscentedKalmanFilter(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=C0103
    """ Implements the Scaled Unscented Kalman filter (UKF) as defined by
    Simon Julier in [1], using the formulation provided by Wan and Merle
    in [2]. This filter scales the sigma points to avoid strong nonlinearities.


    You will have to set the following attributes after constructing this
    object for the filter to perform properly.

    Attributes
    ----------

    x : numpy.array(dim_x)
        state estimate vector

    P : numpy.array(dim_x, dim_x)
        covariance estimate matrix

    R : numpy.array(dim_z, dim_z)
        measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        process noise matrix


    You may read the following attributes.

    Readable Attributes
    -------------------

    xp : numpy.array(dim_x)
        predicted state (result of predict())

    Pp : numpy.array(dim_x, dim_x)
        predicted covariance matrix (result of predict())


    References
    ----------

    .. [1] Julier, Simon J. "The scaled unscented transformation,"
        American Control Converence, 2002, pp 4555-4559, vol 6.

        Online copy:
        https://www.cs.unc.edu/~welch/kalman/media/pdf/ACC02-IEEE1357.PDF


    .. [2] E. A. Wan and R. Van der Merwe, “The unscented Kalman filter for
        nonlinear estimation,” in Proc. Symp. Adaptive Syst. Signal
        Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

        Online Copy:
        https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    """

    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 sqrt_method=cholesky):
        """ Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        Parameters
        ----------

        dim_x : int
            Number of state variables for the filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.


        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        dt : float
            Time between steps in seconds.

        hx : function(x)
            Measurement function. Converts state vector x into a measurement
            vector of shape (dim_z).

        fx : function(x,dt)
            function that returns the state x transformed by the
            state transistion function. dt is the time step in seconds.

        alpha : float
            Determins the spread of the sigma points around the mean.
            Usually a small positive value (1e-3) according to [3].

        beta : float
            Incorporates prior knowledge of the distribution of the mean. For
            Gaussian x beta=2 is optimal, according to [3].

        kappa : float, default=0.0
            Secondary scaling parameter usually set to 0 according to [4],
            or to 3-n according to [5].

        sqrt_method : function(ndarray), default = scipy.linalg.cholesky
            Defines how we compute the square root of a matrix, which has
            no unique answer. Cholesky is the default choice due to its
            speed. Typically your alternative choice will be
            scipy.linalg.sqrtm. Different choices affect how the sigma points
            are arranged relative to the eigenvectors of the covariance matrix.
            Usually this will not matter to you; if so the default cholesky()
            yields maximal performance. As of van der Merwe's dissertation of
            2004 [6] this was not a well reseached area so I have no advice
            to give you.

            If your method returns a triangular matrix it must be upper
            triangular. Do not use numpy.linalg.cholesky - for historical
            reasons it returns a lower triangular matrix. The SciPy version
            does the right thing.


        References
        ----------

        .. [3] S. Julier, J. Uhlmann, and H. Durrant-Whyte. "A new method for
               the nonlinear transformation of means and covariances in filters
               and estimators," IEEE Transactions on Automatic Control, 45(3),
               pp. 477-482 (March 2000).

        .. [4] E. A. Wan and R. Van der Merwe, “The Unscented Kalman filter for
               Nonlinear Estimation,” in Proc. Symp. Adaptive Syst. Signal
               Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

               https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

        .. [5] Wan, Merle "The Unscented Kalman Filter," chapter in *Kalman
               Filtering and Neural Networks*, John Wiley & Sons, Inc., 2001.

        .. [6] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
               Inference in Dynamic State-Space Models" (Doctoral dissertation)
        """

        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.xp = None
        self.Pp = None
        self._dim_x = dim_x
        self._dim_z = dim_z
        self._dt = dt
        self._num_sigmas = 2*dim_x + 1
        #self.alpha = alpha
        #self.beta = beta
        #self.kappa = kappa
        self.hx = hx
        self.fx = fx
        self.sigma_points = points
        self.msqrt = sqrt_method

        # weights for the means and covariances. In this formation
        # both are the same.
        self.Wm, self.Wc = self.sigma_points.weights()


        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update
        self.sigmas_f = zeros((2*self._dim_x+1, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))



    def predict(self, dt=None,  UT=None, fx_args=()):
        """ Performs the predict step of the UKF. On return, self.xp and
        self.Pp contain the predicted state (xp) and covariance (Pp). 'p'
        stands for prediction.

        Important: this MUST be called before update() is called for the first
        time.

        Parameters
        ----------

        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work, but if for example you are using angles the default method
            of computing means and residuals will not work, and you will have
            to define how to compute it.

        fx_args : tuple, optional, default (,)
            optional arguments to be passed into fx() after the required state
            variable.

        """
        if dt is None:
            dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        if UT is None:
            UT = unscented_transform

        # rename for readability
        Wm = self.Wm
        Wc = self.Wc

        # calculate sigma points for given mean and covariance
        sigmas = self.sigma_points.sigma_points(self.x, self.P)

        for i in range(self._num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], dt, *fx_args)

        self.xp, self.Pp = UT(self.sigmas_f, Wm, Wc, self.Q)


    def update(self, z, R=None, UT=unscented_transform, hx_args=(),
               residual_x=np.subtract, residual_h=np.subtract):
        """ Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        **Parameters**

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work, but if for example you are using angles the default method
            of computing means and residuals will not work, and you will have
            to define how to compute it.

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.

        residual_x : function (x, x2), optional
            Optional function that computes the residual (difference) between
            the two state vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)

        residual_h : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used.
        """

        if z is None:
            return

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        for i in range(self._num_sigmas):
            self.sigmas_h[i] = self.hx(self.sigmas_f[i], *hx_args)


        # mean and covariance of prediction passed through unscented transform
        zp, Pz = UT(self.sigmas_h, self.Wm, self.Wc, R)

        # compute cross variance of the state and the measurements
        Pxz = zeros((self._dim_x, self._dim_z))
        for i in range(self._num_sigmas):
            Pxz += self.Wc[i] * outer(residual_x(self.sigmas_f[i], self.xp),
                                      residual_h(self.sigmas_h[i], zp))

        K = dot(Pxz, inv(Pz))   # Kalman gain
        y = residual_h(z, zp)   #residual

        self.x = self.xp + dot(K, y)
        self.P = self.Pp - dot3(K, Pz, K.T)


    def batch_filter(self, zs, Rs=None, residual=np.subtract, UT=None):
        """ Performs the UKF filter over the list of measurement in `zs`.


        **Parameters**

        zs : list-like
            list of measurements at each time step `self._dt` Missing
            measurements must be represented by 'None'.

        Rs : list-like, optional
            optional list of values to use for the measurement error
            covariance; a value of None in any position will cause the filter
            to use `self.R` for that time step.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work, but if for example you are using angles the default method
            of computing means and residuals will not work, and you will have
            to define how to compute it.

        **Returns**

        means: ndarray((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance: ndarray((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        """

        try:
            z = zs[0]
        except:
            assert not isscalar(zs), 'zs must be list-like'

        if self._dim_z == 1:
            assert isscalar(z) or (z.ndim==1 and len(z) == 1), \
            'zs must be a list of scalars or 1D, 1 element arrays'

        else:
            assert len(z) == self._dim_z, 'each element in zs must be a' \
            '1D array of length {}'.format(self._dim_z)

        z_n = np.size(zs, 0)
        if Rs is None:
            Rs = [None] * z_n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((z_n, self._dim_x))
        else:
            means = zeros((z_n, self._dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r) in enumerate(zip(zs, Rs)):
            self.predict()
            self.update(z, r)
            means[i,:]         = self.x
            covariances[i,:,:] = self.P

        return (means, covariances)


    def rts_smoother(self, Xs, Ps, Qs=None, dt=None):
        """ Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.

        **Parameters**

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Q : list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        dt : optional, float or array-like of float
            If provided, specifies the time step of each step of the filter.
            If float, then the same time step is used for all steps. If
            an array, then each element k contains the time  at step k.
            Units are seconds.

        **Returns**

        'x' : numpy.ndarray
           smoothed means

        'P' : numpy.ndarray
           smoothed state covariances

        'K' : numpy.ndarray
            smoother gain at each step


        **Example**::

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K) = rts_smoother(mu, cov, fk.F, fk.Q)

        """
        assert len(Xs) == len(Ps)
        n, dim_x = Xs.shape

        if dt is None:
            dt = [self._dt] * n
        elif isscalar(dt):
            dt = [dt] * n

        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        Ks = zeros((n,dim_x,dim_x))

        num_sigmas = 2*dim_x + 1

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f = zeros((num_sigmas, dim_x))


        for k in range(n-2,-1,-1):
            # create sigma points from state estimate, pass through state func
            sigmas = self.sigma_points.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[i] = self.fx(sigmas[i], dt[k])

            # compute backwards prior state and covariance
            xb = dot(self.Wm, sigmas_f)
            Pb = 0
            x = Xs[k]
            for i in range(num_sigmas):
                y = sigmas_f[i] - x
                Pb += self.Wm[i] * outer(y, y)
            Pb += Qs[k]

            # compute cross variance
            Pxb = 0
            for i in range(num_sigmas):
                z = sigmas[i] - Xs[k]
                y = sigmas_f[i] - xb
                Pxb += self.Wm[i] * outer(z, y)

            # compute gain
            K = dot(Pxb, inv(Pb))

            # update the smoothed estimates
            xs[k] += dot (K, xs[k+1] - xb)
            ps[k] += dot3(K, ps[k+1] - Pb, K.T)
            Ks[k] = K

        return (xs, ps, Ks)
