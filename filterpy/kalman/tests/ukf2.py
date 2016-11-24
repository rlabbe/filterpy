# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
6"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import matplotlib.pyplot as plt
import numpy.random as random
from numpy.random import randn
from numpy import asarray
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import (unscented_transform, MerweScaledSigmaPoints,
                             JulierSigmaPoints, SimplexSigmaPoints)
from filterpy.common import Q_discrete_white_noise
import filterpy.stats as stats
from math import cos, sin



class UnscentedKalmanFilter2(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=C0103
    r""" Implements the Scaled Unscented Kalman filter (UKF) as defined by
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


    K : numpy.array
        Kalman gain

    y : numpy.array
        innovation residual

    x : numpy.array(dim_x)
        predicted/updated state (result of predict()/update())

    P : numpy.array(dim_x, dim_x)
        predicted/updated covariance matrix (result of predict()/update())

    likelihood : scalar
        Likelihood of last measurement update.

    log_likelihood : scalar
        Log likelihood of last measurement update.


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
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None):
        r""" Create a Kalman filter. You are responsible for setting the
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

        points : class
            Class which computes the sigma points and weights for a UKF
            algorithm. You can vary the UKF implementation by changing this
            class. For example, MerweScaledSigmaPoints implements the alpha,
            beta, kappa parameterization of Van der Merwe, and
            JulierSigmaPoints implements Julier's original kappa
            parameterization. See either of those for the required
            signature of this class if you want to implement your own.

        sqrt_fn : callable(ndarray), default = scipy.linalg.cholesky
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

        x_mean_fn : callable  (sigma_points, weights), optional
            Function that computes the mean of the provided sigma points
            and weights. Use this if your state variable contains nonlinear
            values such as angles which cannot be summed.

            .. code-block:: Python

                def state_mean(sigmas, Wm):
                    x = np.zeros(3)
                    sum_sin, sum_cos = 0., 0.

                    for i in range(len(sigmas)):
                        s = sigmas[i]
                        x[0] += s[0] * Wm[i]
                        x[1] += s[1] * Wm[i]
                        sum_sin += sin(s[2])*Wm[i]
                        sum_cos += cos(s[2])*Wm[i]
                    x[2] = atan2(sum_sin, sum_cos)
                    return x

        z_mean_fn : callable  (sigma_points, weights), optional
            Same as x_mean_fn, except it is called for sigma points which
            form the measurements after being passed through hx().

        residual_x : callable (x, y), optional
        residual_z : callable (x, y), optional
            Function that computes the residual (difference) between x and y.
            You will have to supply this if your state variable cannot support
            subtraction, such as angles (359-1 degreees is 2, not 358). x and y
            are state vectors, not scalars. One is for the state variable,
            the other is for the measurement state.

            .. code-block:: Python

                def residual(a, b):
                    y = a[0] - b[0]
                    if y > np.pi:
                        y -= 2*np.pi
                    if y < -np.pi:
                        y = 2*np.pi
                    return y


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
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn
        self.log_likelihood = 0.0

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        self.Wm, self.Wc = self.points_fn.weights()
        self.Wm = self.Wm.reshape((-1, 1))
        self.Wc = self.Wc.reshape((-1, 1))

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = zeros((self._dim_x, self._num_sigmas))
        self.sigmas_h = zeros((self._dim_z, self._num_sigmas))


    def predict(self, dt=None,  UT=None, fx_args=()):
        r""" Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '

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
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        fx_args : tuple, optional, default (,)
            optional arguments to be passed into fx() after the required state
            variable.
        """

        if dt is None:
            dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        if UT is None:
            UT = unscented_transform2

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i in range(self._num_sigmas):
            self.sigmas_f[:, i] = self.fx(sigmas[:, i], dt, *fx_args)

        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
                            self.x_mean, self.residual_x)


    def update(self, z, R=None, UT=None, hx_args=()):
        """ Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.
        """

        if z is None:
            return

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if UT is None:
            UT = unscented_transform2

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        for i in range(self._num_sigmas):
            self.sigmas_h[:, i] = self.hx(self.sigmas_f[:, i], *hx_args)

        # mean and covariance of prediction passed through unscented transform
        zp, Pz = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)

        # compute cross variance of the state and the measurements
        Pxz = zeros((self._dim_x, self._dim_z))
        for i in range(self._num_sigmas):
            dx = self.residual_x(self.sigmas_f[:, i:i+1], self.x)
            dz =  self.residual_z(self.sigmas_h[:, i:i+1], zp)
            Pxz += self.Wc[i] * outer(dx, dz)


        self.K = dot(Pxz, inv(Pz))        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot3(self.K, Pz, self.K.T)

        self.log_likelihood = multivariate_normal.logpdf(
            x=self.y, mean=np.zeros(len(self.y)), cov=Pz, allow_singular=True)


    @property
    def likelihood(self):
        return math.exp(self.log_likelihood)


    def batch_filter(self, zs, Rs=None, UT=None):
        """ Performs the UKF filter over the list of measurement in `zs`.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self._dt` Missing
            measurements must be represented by 'None'.

        Rs : list-like, optional
            optional list of values to use for the measurement error
            covariance; a value of None in any position will cause the filter
            to use `self.R` for that time step.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        Returns
        -------

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
        means = zeros((z_n, self._dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r) in enumerate(zip(zs, Rs)):
            self.predict(UT=UT)
            self.update(z, r, UT=UT)
            means[i,:]         = self.x
            covariances[i,:,:] = self.P

        return (means, covariances)


    def rts_smoother(self, Xs, Ps, Qs=None, dt=None):
        """ Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Qs: list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        dt : optional, float or array-like of float
            If provided, specifies the time step of each step of the filter.
            If float, then the same time step is used for all steps. If
            an array, then each element k contains the time  at step k.
            Units are seconds.

        Returns
        -------

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K) = rts_smoother(mu, cov, fk.F, fk.Q)
        """

        assert len(Xs) == len(Ps)
        n, dim_x, _ = Xs.shape

        if dt is None:
            dt = [self._dt] * n
        elif isscalar(dt):
            dt = [dt] * n

        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        Ks = zeros((n,dim_x,dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f = zeros((dim_x, num_sigmas))

        for k in range(n-2,-1,-1):
            # create sigma points from state estimate, pass through state func
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[:, i] = self.fx(sigmas[:, i], dt[k])

            # compute backwards prior state and covariance
            xb = dot(sigmas_f, self.Wm)
            Pb = 0
            x = Xs[k]
            for i in range(num_sigmas):
                y = self.residual_x(sigmas_f[:, i:i+1], x)
                Pb += self.Wc[i] * outer(y, y)
            Pb += Qs[k]

            # compute cross variance
            Pxb = 0
            for i in range(num_sigmas):
                z = self.residual_x(sigmas[:,i:i+1], Xs[k])
                y = self.residual_x(sigmas_f[:,i:i+1], xb)
                Pxb += self.Wc[i] * outer(z, y)

            # compute gain
            K = dot(Pxb, inv(Pb))

            # update the smoothed estimates
            xs[k] += dot (K, self.residual_x(xs[k+1], xb))
            ps[k] += dot3(K, ps[k+1] - Pb, K.T)
            Ks[k] = K

        return (xs, ps, Ks)

  
class MerweScaledSigmaPoints2(object):
 
    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        """ Generates sigma points and weights according to Van der Merwe's
        2004 dissertation[1]. It parametizes the sigma points using
        alpha, beta, kappa terms, and is the version seen in most publications.
 
        Unless you know better, this should be your default choice.
 
        Parameters
        ----------
 
        n : int
            Dimensionality of the state. 2n+1 weights will be generated.
 
        alpha : float
            Determins the spread of the sigma points around the mean.
            Usually a small positive value (1e-3) according to [3].
 
        beta : float
            Incorporates prior knowledge of the distribution of the mean. For
            Gaussian x beta=2 is optimal, according to [3].
 
        kappa : float, default=0.0
            Secondary scaling parameter usually set to 0 according to [4],
            or to 3-n according to [5].
 
        sqrt_method : function(ndarray), default=scipy.linalg.cholesky
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
 
        subtract : callable (x, y), optional
            Function that computes the difference between x and y.
            You will have to supply this if your state variable cannot support
            subtraction, such as angles (359-1 degreees is 2, not 358). x and y
            are state vectors, not scalars.
 
        References
        ----------
 
        .. [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
               Inference in Dynamic State-Space Models" (Doctoral dissertation)
 
        """
 
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method
 
        if subtract is None:
            self.subtract= np.subtract
        else:
            self.subtract = subtract
 
 
    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1
 
 
    def sigma_points(self, x, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.
 
        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
 
        Parameters
        ----------
 
        X An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])
 
        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.
 
        Returns
        -------
 
        sigmas : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.
 
            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """
 
 
        assert x.ndim == 2 and x.shape[1], "x must be a column vector"
 
        n = self.n
 
        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.asarray(P)
 
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n)*P)
 
        sigmas = np.zeros((n, 2*n+1))
        x0 = x[:, 0]
        sigmas[:,0] = x0
        for k in range(n):
            sigmas[:, k+1]   = self.subtract(x0, -U[k])
            sigmas[:, n+k+1] = self.subtract(x0, U[k])
 
        return sigmas
 
 
    def weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.
 
        Returns
        -------
 
        Wm : ndarray[2n+1]
            weights for mean
 
        Wc : ndarray[2n+1]
            weights for the covariances
        """
 
        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n
 
        c = .5 / (n + lambda_)
        Wc = np.full(2*n + 1, c)
        Wm = np.full(2*n + 1, c)
        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        Wm[0] = lambda_ / (n + lambda_)
 
        return Wm, Wc
 
 def unscented_transform2(sigmas, Wm, Wc, noise_cov=None,
                        mean_fn=np.dot, residual_fn=None):
    """ Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.

    Parameters
    ----------

    sigmas: ndarray [#sigmas per dimension, dimension]
        2D array of sigma points.

    Wm : ndarray [# sigmas per dimension]
        Weights for the mean. Must sum to 1.


    Wc : ndarray [# sigmas per dimension]
        Weights for the covariance. Must sum to 1.

    noise_cov : ndarray, optional
        noise matrix added to the final computed covariance matrix.

    mean_fn : callable (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this if your state variable contains nonlinear
        values such as angles which cannot be summed.

        .. code-block:: Python

            def state_mean(sigmas, Wm):
                x = np.zeros(3)
                sum_sin, sum_cos = 0., 0.

                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += sin(s[2])*Wm[i]
                    sum_cos += cos(s[2])*Wm[i]
                x[2] = atan2(sum_sin, sum_cos)
                return x

    residual_fn : callable (x, y), optional

        Function that computes the residual (difference) between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.

        .. code-block:: Python

            def residual(a, b):
                y = a[0] - b[0]
                if y > np.pi:
                    y -= 2*np.pi
                if y < -np.pi:
                    y = 2*np.pi
                return y


    Returns
    -------

    x : ndarray [dimension]
        Mean of the sigma points after passing through the transform.

    P : ndarray
        covariance of the sigma points after passing throgh the transform.
    """

    n, kmax = sigmas.shape


    x = np.dot(sigmas, Wm).reshape((-1, 1))   # dot = \Sigma^n_1 (W[k]*Xi[k])


    # new covariance is the sum of the outer product of the residuals
    # times the weights

    # this is the fast way to do this - see 'else' for the slow way
    if residual_fn is None:
        y = sigmas - x[np.newaxis,:]
        P = y.T.dot(np.diag(Wc)).dot(y)
    else:
        P = np.zeros((n, n))
        for k in range(kmax):
            y = residual_fn(sigmas[:, k], x[:, 0])
            P += Wc[k] * np.outer(y, y)

    if noise_cov is not None:
        P += noise_cov

    return (x, P)


DO_PLOT = False


def test_sigma_plot():
    """ Test to make sure sigma's correctly mirror the shape and orientation
    of the covariance array."""

    x = np.array([[1, 2]])
    P = np.array([[2, 1.2],
                  [1.2, 2]])
    kappa = .1

    # if kappa is larger, than points shoudld be closer together

    sp0 = JulierSigmaPoints(n=2, kappa=kappa)
    sp1 = JulierSigmaPoints(n=2, kappa=kappa*1000)
    sp2 = MerweScaledSigmaPoints(n=2, kappa=0, beta=2, alpha=1e-3)
    sp3 = SimplexSigmaPoints(n=2)

    w0, _ = sp0.weights()
    w1, _ = sp1.weights()
    w2, _ = sp2.weights()
    w3, _ = sp3.weights()

    Xi0 = sp0.sigma_points(x, P)
    Xi1 = sp1.sigma_points(x, P)
    Xi2 = sp2.sigma_points(x, P)
    Xi3 = sp3.sigma_points(x, P)

    assert max(Xi1[:,0]) > max(Xi0[:,0])
    assert max(Xi1[:,1]) > max(Xi0[:,1])

    if DO_PLOT:
        plt.figure()
        for i in range(Xi0.shape[0]):
            plt.scatter((Xi0[i,0]-x[0, 0])*w0[i] + x[0, 0],
                        (Xi0[i,1]-x[0, 1])*w0[i] + x[0, 1],
                         color='blue', label='Julier low $\kappa$')

        for i in range(Xi1.shape[0]):
            plt.scatter((Xi1[i, 0]-x[0, 0]) * w1[i] + x[0,0],
                        (Xi1[i, 1]-x[0, 1]) * w1[i] + x[0,1],
                         color='green', label='Julier high $\kappa$')
        # for i in range(Xi2.shape[0]):
        #     plt.scatter((Xi2[i, 0] - x[0, 0]) * w2[i] + x[0, 0],
        #                 (Xi2[i, 1] - x[0, 1]) * w2[i] + x[0, 1],
        #                 color='red')
        for i in range(Xi3.shape[0]):
            plt.scatter((Xi3[i, 0] - x[0, 0]) * w3[i] + x[0, 0],
                        (Xi3[i, 1] - x[0, 1]) * w3[i] + x[0, 1],
                        color='black', label='Simplex')

        stats.plot_covariance_ellipse([1, 2], P)


def test_simplex_weights():
    for n in range(1,15):
        for k in np.linspace(0,5,0.1):
            Wm = UKF.weights(n, k)

            assert abs(sum(Wm) - 1) < 1.e-12


def test_julier_weights():
    for n in range(1,15):
        for k in np.linspace(0,5,0.1):
            Wm = UKF.weights(n, k)

            assert abs(sum(Wm) - 1) < 1.e-12


def test_scaled_weights():
    for n in range(1,5):
        for alpha in np.linspace(0.99, 1.01, 100):
            for beta in range(0,2):
                for kappa in range(0,2):
                    sp = MerweScaledSigmaPoints(n, alpha, 0, 3-n)
                    Wm, Wc = sp.weights()
                    assert abs(sum(Wm) - 1) < 1.e-1
                    assert abs(sum(Wc) - 1) < 1.e-1


def test_julier_sigma_points_1D():
    """ tests passing 1D data into sigma_points"""

    kappa = 0.
    sp = JulierSigmaPoints(1, kappa)

    #ukf = UKF(dim_x=1, dim_z=1, dt=0.1, hx=None, fx=None, kappa=kappa)

    Wm, Wc = sp.weights()
    assert np.allclose(Wm, Wc, 1e-12)
    assert len(Wm) == 3

    mean = 5
    cov = 9

    Xi = sp.sigma_points(mean, cov)
    xm, ucov = unscented_transform(Xi, Wm, Wc, 0)

    # sum of weights*sigma points should be the original mean
    m = 0.0
    for x, w in zip(Xi, Wm):
        m += x*w

    assert abs(m-mean) < 1.e-12
    assert abs(xm[0] - mean) < 1.e-12
    assert abs(ucov[0,0]-cov) < 1.e-12

    assert Xi.shape == (3,1)


def test_simplex_sigma_points_1D():
    """ tests passing 1D data into sigma_points"""

    sp = SimplexSigmaPoints(1)

    #ukf = UKF(dim_x=1, dim_z=1, dt=0.1, hx=None, fx=None, kappa=kappa)

    Wm, Wc = sp.weights()
    assert np.allclose(Wm, Wc, 1e-12)
    assert len(Wm) == 2

    mean = 5
    cov = 9

    Xi = sp.sigma_points(mean, cov)
    xm, ucov = unscented_transform(Xi, Wm, Wc, 0)

    # sum of weights*sigma points should be the original mean
    m = 0.0
    for x, w in zip(Xi, Wm):
        m += x*w

    assert abs(m-mean) < 1.e-12
    assert abs(xm[0] - mean) < 1.e-12
    assert abs(ucov[0,0]-cov) < 1.e-12

    assert Xi.shape == (2,1)


class RadarSim(object):
    def __init__(self, dt):
        self.x = 0
        self.dt = dt

    def get_range(self):
        vel = 100  + 5*randn()
        alt = 1000 + 10*randn()
        self.x += vel*self.dt

        v = self.x * 0.05*randn()
        rng = (self.x**2 + alt**2)**.5 + v
        return rng


def test_radar():
    def fx(x, dt):
        A = np.eye(3) + dt * np.array ([[0, 1, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]])
        return A.dot(x)

    def hx(x):
        return np.sqrt(x[0]**2 + x[2]**2)

    dt = 0.05

    sp = JulierSigmaPoints(n=3, kappa=0.)
    # sp = SimplexSigmaPoints(n=3)
    kf = UKF(3, 1, dt, fx=fx, hx=hx, points=sp)

    kf.Q *= 0.01
    kf.R = 10
    kf.x = np.array([0., 90., 1100.])
    kf.P *= 100.
    radar = RadarSim(dt)

    t = np.arange(0,20+dt, dt)

    n = len(t)

    xs = np.zeros((n,3))

    random.seed(200)
    rs = []
    #xs = []
    for i in range(len(t)):
        r = radar.get_range()
        #r = GetRadar(dt)
        kf.predict()
        kf.update(z=[r])

        xs[i,:] = kf.x
        rs.append(r)

    if DO_PLOT:
        print(xs[:,0].shape)

        plt.figure()
        plt.subplot(311)
        plt.plot(t, xs[:,0])
        plt.subplot(312)
        plt.plot(t, xs[:,1])
        plt.subplot(313)
        plt.plot(t, xs[:,2])


def test_linear_2d_merwe():
    """ should work like a linear KF if problem is linear """


    def fx(x, dt):
        F = np.array([[1, dt, 0, 0],
                      [0,  1, 0, 0],
                      [0, 0,  1, dt],
                      [0, 0, 0,  1]], dtype=float)

        return np.dot(F, x)

    def hx(x):
        return np.array([x[0], x[2]])


    dt = 0.1
    points = MerweScaledSigmaPoints(4, .1, 2., -1)
    kf = UKF(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)


    kf.x = np.array([-1., 1., -1., 1])
    kf.P*=0.0001
    #kf.R *=0
    #kf.Q

    zs = []
    for i in range(20):
        z = np.array([i+randn()*0.1, i+randn()*0.1])
        zs.append(z)

    Ms, Ps = kf.batch_filter(zs)
    smooth_x, _, _ = kf.rts_smoother(Ms, Ps, dt=dt)

    if DO_PLOT:
        plt.figure()
        zs = np.asarray(zs)
        plt.plot(zs[:,0], marker='+')
        plt.plot(Ms[:,0], c='b')
        plt.plot(smooth_x[:,0], smooth_x[:,2], c='r')
        print(smooth_x)

from filterpy.kalman import UnscentedKalmanFilter2 as UKF2

def test_linear_2d_merwe_column():
    """ should work like a linear KF if problem is linear """


    def fx(x, dt):
        F = np.array([[1, dt, 0, 0],
                      [0,  1, 0, 0],
                      [0, 0,  1, dt],
                      [0, 0, 0,  1]], dtype=float)

        return np.dot(F, x)

    def hx(x):
        return np.array([x[0], x[2]])


    dt = 0.1
    points = MerweScaledSigmaPoints2(4, .1, 2., -1)
    kf = UKF2(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)


    kf.x = np.array([[-1., 1., -1., 1]]).T
    kf.P*=0.0001
    #kf.R *=0
    #kf.Q

    zs = []
    for i in range(20):
        z = np.array([[i+randn()*0.1],
                      [i+randn()*0.1]])
        zs.append(z)

    Ms, Ps = kf.batch_filter(zs)
    smooth_x, _, _ = kf.rts_smoother(Ms, Ps, dt=dt)

    if DO_PLOT:
        plt.figure()
        zs = np.asarray(zs)
        plt.plot(zs[:,0], marker='+', c='b')
        plt.plot(Ms[:,0], c='b')
        plt.plot(smooth_x[:,0], smooth_x[:,2], c='r')
        print(smooth_x)

def test_linear_2d_simplex():
    """ should work like a linear KF if problem is linear """


    def fx(x, dt):
        F = np.array([[1, dt, 0, 0],
                      [0,  1, 0, 0],
                      [0, 0,  1, dt],
                      [0, 0, 0,  1]], dtype=float)

        return np.dot(F, x)

    def hx(x):
        return np.array([x[0], x[2]])


    dt = 0.1
    points = SimplexSigmaPoints(n=4)
    kf = UKF(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)


    kf.x = np.array([-1., 1., -1., 1])
    kf.P*=0.0001
    #kf.R *=0
    #kf.Q

    zs = []
    for i in range(20):
        z = np.array([i+randn()*0.1, i+randn()*0.1])
        zs.append(z)

    Ms, Ps = kf.batch_filter(zs)
    smooth_x, _, _ = kf.rts_smoother(Ms, Ps, dt=dt)

    if DO_PLOT:
        zs = np.asarray(zs)

        #plt.plot(zs[:,0])
        plt.plot(Ms[:,0])
        plt.plot(smooth_x[:,0], smooth_x[:,2])

        print(smooth_x)

def test_linear_1d():
    """ should work like a linear KF if problem is linear """


    def fx(x, dt):
        F = np.array([[1., dt],
                      [0,  1]], dtype=float)

        return np.dot(F, x)

    def hx(x):
        return np.array([x[0]])


    dt = 0.1
    points = MerweScaledSigmaPoints(2, .1, 2., -1)
    kf = UKF(dim_x=2, dim_z=1, dt=dt, fx=fx, hx=hx, points=points)


    kf.x = np.array([1, 2])
    kf.P = np.array([[1, 1.1],
                     [1.1, 3]])
    kf.R *= 0.05
    kf.Q = np.array([[0., 0], [0., .001]])

    z = np.array([2.])
    kf.predict()
    kf.update(z)

    zs = []
    for i in range(50):
        z = np.array([i+randn()*0.1])
        zs.append(z)

        kf.predict()
        kf.update(z)
        print('K', kf.K.T)
        print('x', kf.x)



def test_batch_missing_data():
    """ batch filter should accept missing data with None in the measurements """


    def fx(x, dt):
        F = np.array([[1, dt, 0, 0],
                      [0,  1, 0, 0],
                      [0, 0,  1, dt],
                      [0, 0, 0,  1]], dtype=float)

        return np.dot(F, x)

    def hx(x):
        return np.array([x[0], x[2]])


    dt = 0.1
    points = MerweScaledSigmaPoints(4, .1, 2., -1)
    kf = UKF(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)


    kf.x = np.array([-1., 1., -1., 1])
    kf.P*=0.0001

    zs = []
    for i in range(20):
        z = np.array([i+randn()*0.1, i+randn()*0.1])
        zs.append(z)

    zs[2] = None
    Rs = [1]*len(zs)
    Rs[2] = None
    Ms, Ps = kf.batch_filter(zs)


def test_rts():
    def fx(x, dt):
        A = np.eye(3) + dt * np.array ([[0, 1, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]])
        f = np.dot(A, x)
        return f

    def hx(x):
        return np.sqrt (x[0]**2 + x[2]**2)

    dt = 0.05

    sp = JulierSigmaPoints(n=3, kappa=1.)
    kf = UKF(3, 1, dt, fx=fx, hx=hx, points=sp)

    kf.Q *= 0.01
    kf.R = 10
    kf.x = np.array([0., 90., 1100.])
    kf.P *= 100.
    radar = RadarSim(dt)

    t = np.arange(0,20+dt, dt)

    n = len(t)

    xs = np.zeros((n,3))

    random.seed(200)
    rs = []
    #xs = []
    for i in range(len(t)):
        r = radar.get_range()
        #r = GetRadar(dt)
        kf.predict()
        kf.update(z=[r])

        xs[i,:] = kf.x
        rs.append(r)


    kf.x = np.array([0., 90., 1100.])
    kf.P = np.eye(3)*100
    M, P = kf.batch_filter(rs)
    assert np.array_equal(M, xs), "Batch filter generated different output"

    Qs = [kf.Q]*len(t)
    M2, P2, K = kf.rts_smoother(Xs=M, Ps=P, Qs=Qs)


    if DO_PLOT:
        print(xs[:,0].shape)

        plt.figure()
        plt.subplot(311)
        plt.plot(t, xs[:,0])
        plt.plot(t, M2[:,0], c='g')
        plt.subplot(312)
        plt.plot(t, xs[:,1])
        plt.plot(t, M2[:,1], c='g')
        plt.subplot(313)

        plt.plot(t, xs[:,2])
        plt.plot(t, M2[:,2], c='g')


def test_fixed_lag():
    def fx(x, dt):
        A = np.eye(3) + dt * np.array ([[0, 1, 0],
                                        [0, 0, 0],
                                        [0, 0, 0]])
        f = np.dot(A, x)
        return f

    def hx(x):
        return np.sqrt (x[0]**2 + x[2]**2)

    dt = 0.05

    sp = JulierSigmaPoints(n=3, kappa=0)

    kf = UKF(3, 1, dt, fx=fx, hx=hx, points=sp)

    kf.Q *= 0.01
    kf.R = 10
    kf.x = np.array([0., 90., 1100.])
    kf.P *= 1.
    radar = RadarSim(dt)

    t = np.arange(0,20+dt, dt)

    n = len(t)

    xs = np.zeros((n,3))

    random.seed(200)
    rs = []
    #xs = []

    M = []
    P = []
    N =10
    flxs = []
    for i in range(len(t)):
        r = radar.get_range()
        #r = GetRadar(dt)
        kf.predict()
        kf.update(z=[r])

        xs[i,:] = kf.x
        flxs.append(kf.x)
        rs.append(r)
        M.append(kf.x)
        P.append(kf.P)
        print(i)
        #print(i, np.asarray(flxs)[:,0])
        if i == 20 and len(M) >= N:
            try:
                M2, P2, K = kf.rts_smoother(Xs=np.asarray(M)[-N:], Ps=np.asarray(P)[-N:])
                flxs[-N:] = M2
                #flxs[-N:] = [20]*N
            except:
                print('except', i)
            #P[-N:] = P2


    kf.x = np.array([0., 90., 1100.])
    kf.P = np.eye(3)*100
    M, P = kf.batch_filter(rs)

    Qs = [kf.Q]*len(t)
    M2, P2, K = kf.rts_smoother(Xs=M, Ps=P, Qs=Qs)


    flxs = np.asarray(flxs)
    print(xs[:,0].shape)

    plt.figure()
    plt.subplot(311)
    plt.plot(t, xs[:,0])
    plt.plot(t, flxs[:,0], c='r')
    plt.plot(t, M2[:,0], c='g')
    plt.subplot(312)
    plt.plot(t, xs[:,1])
    plt.plot(t, flxs[:,1], c='r')
    plt.plot(t, M2[:,1], c='g')

    plt.subplot(313)
    plt.plot(t, xs[:,2])
    plt.plot(t, flxs[:,2], c='r')
    plt.plot(t, M2[:,2], c='g')



def test_circle():
    from filterpy.kalman import KalmanFilter
    from math import radians
    def hx(x):
        radius = x[0]
        angle = x[1]
        x = cos(radians(angle)) * radius
        y = sin(radians(angle)) * radius
        return np.array([x, y])

    def fx(x, dt):
        return np.array([x[0], x[1]+x[2], x[2]])

    std_noise = .1

    sp = JulierSigmaPoints(n=3, kappa=0.)
    f = UKF(dim_x=3, dim_z=2, dt=.01, hx=hx, fx=fx, points=sp)
    f.x = np.array([50., 90., 0])
    f.P *= 100
    f.R = np.eye(2)*(std_noise**2)
    f.Q = np.eye(3)*.001
    f.Q[0,0]=0
    f.Q[2,2]=0

    kf = KalmanFilter(dim_x=6, dim_z=2)
    kf.x = np.array([50., 0., 0, 0, .0, 0.])

    F = np.array([[1., 1., .5, 0., 0., 0.],
                  [0., 1., 1., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 1., 1., .5],
                  [0., 0., 0., 0., 1., 1.],
                  [0., 0., 0., 0., 0., 1.]])

    kf.F = F
    kf.P*= 100
    kf.H = np.array([[1,0,0,0,0,0],
                     [0,0,0,1,0,0]])


    kf.R = f.R
    kf.Q[0:3, 0:3] = Q_discrete_white_noise(3, 1., .00001)
    kf.Q[3:6, 3:6] = Q_discrete_white_noise(3, 1., .00001)

    measurements = []
    results = []

    zs = []
    kfxs = []
    for t in range (0,12000):
        a = t / 30 + 90
        x = cos(radians(a)) * 50.+ randn() * std_noise
        y = sin(radians(a)) * 50. + randn() * std_noise
        # create measurement = t plus white noise
        z = np.array([x,y])
        zs.append(z)

        f.predict()
        f.update(z)

        kf.predict()
        kf.update(z)

        # save data
        results.append (hx(f.x))
        kfxs.append(kf.x)
        #print(f.x)

    results = np.asarray(results)
    zs = np.asarray(zs)
    kfxs = np.asarray(kfxs)

    print(results)
    if DO_PLOT:
        plt.plot(zs[:,0], zs[:,1], c='r', label='z')
        plt.plot(results[:,0], results[:,1], c='k', label='UKF')
        plt.plot(kfxs[:,0], kfxs[:,3], c='g', label='KF')
        plt.legend(loc='best')
        plt.axis('equal')



def kf_circle():
    from filterpy.kalman import KalmanFilter
    from math import radians
    import math
    def hx(x):
        radius = x[0]
        angle = x[1]
        x = cos(radians(angle)) * radius
        y = sin(radians(angle)) * radius
        return np.array([x, y])

    def fx(x, dt):
        return np.array([x[0], x[1]+x[2], x[2]])


    def hx_inv(x, y):
        angle = math.atan2(y,x)
        radius = math.sqrt(x*x + y*y)
        return np.array([radius, angle])


    std_noise = .1


    kf = KalmanFilter(dim_x=3, dim_z=2)
    kf.x = np.array([50., 0., 0.])

    F = np.array([[1., 0, 0.],
                  [0., 1., 1.,],
                  [0., 0., 1.,]])

    kf.F = F
    kf.P*= 100
    kf.H = np.array([[1,0,0],
                     [0,1,0]])

    kf.R = np.eye(2)*(std_noise**2)
    #kf.Q[0:3, 0:3] = Q_discrete_white_noise(3, 1., .00001)



    zs = []
    kfxs = []
    for t in range (0,2000):
        a = t / 30 + 90
        x = cos(radians(a)) * 50.+ randn() * std_noise
        y = sin(radians(a)) * 50. + randn() * std_noise

        z = hx_inv(x,y)
        zs.append(z)

        kf.predict()
        kf.update(z)

        # save data
        kfxs.append(kf.x)

    zs = np.asarray(zs)
    kfxs = np.asarray(kfxs)


    if DO_PLOT:
        plt.plot(zs[:,0], zs[:,1], c='r', label='z')
        plt.plot(kfxs[:,0], kfxs[:,1], c='g', label='KF')
        plt.legend(loc='best')
        plt.axis('equal')




def two_radar():

    # code is not complete - I was using to test RTS smoother. very similar
    # to two_radary.py in book.

    import numpy as np
    import matplotlib.pyplot as plt

    from numpy import array
    from numpy.linalg import norm
    from numpy.random import randn
    from math import atan2, radians

    from filterpy.common import Q_discrete_white_noise

    class RadarStation(object):

        def __init__(self, pos, range_std, bearing_std):
            self.pos = asarray(pos)

            self.range_std = range_std
            self.bearing_std = bearing_std


        def reading_of(self, ac_pos):
            """ Returns range and bearing to aircraft as tuple. bearing is in
            radians.
            """

            diff = np.subtract(self.pos, ac_pos)
            rng = norm(diff)
            brg = atan2(diff[1], diff[0])
            return rng, brg


        def noisy_reading(self, ac_pos):
            rng, brg = self.reading_of(ac_pos)
            rng += randn() * self.range_std
            brg += randn() * self.bearing_std
            return rng, brg




    class ACSim(object):

        def __init__(self, pos, vel, vel_std):
            self.pos = asarray(pos, dtype=float)
            self.vel = asarray(vel, dtype=float)
            self.vel_std = vel_std


        def update(self):
            vel = self.vel + (randn() * self.vel_std)
            self.pos += vel

            return self.pos

    dt = 1.


    def hx(x):
        r1, b1 = hx.R1.reading_of((x[0], x[2]))
        r2, b2 = hx.R2.reading_of((x[0], x[2]))

        return array([r1, b1, r2, b2])
        pass



    def fx(x, dt):
        x_est = x.copy()
        x_est[0] += x[1]*dt
        x_est[2] += x[3]*dt
        return x_est



    vx, vy = 0.1, 0.1

    f = UnscentedKalmanFilter(dim_x=4, dim_z=4, dt=dt, hx=hx, fx=fx, kappa=0)
    aircraft = ACSim ((100,100), (vx*dt,vy*dt), 0.00000002)


    range_std = 0.001  # 1 meter
    bearing_std = 1/1000 # 1mrad

    R1 = RadarStation ((0,0), range_std, bearing_std)
    R2 = RadarStation ((200,0), range_std, bearing_std)

    hx.R1 = R1
    hx.R2 = R2

    f.x = array([100, vx, 100, vy])

    f.R = np.diag([range_std**2, bearing_std**2, range_std**2, bearing_std**2])
    q = Q_discrete_white_noise(2, var=0.0002, dt=dt)
    #q = np.array([[0,0],[0,0.0002]])
    f.Q[0:2, 0:2] = q
    f.Q[2:4, 2:4] = q
    f.P = np.diag([.1, 0.01, .1, 0.01])


    track = []
    zs = []


    for i in range(int(300/dt)):

        pos = aircraft.update()

        r1, b1 = R1.noisy_reading(pos)
        r2, b2 = R2.noisy_reading(pos)

        z = np.array([r1, b1, r2, b2])
        zs.append(z)
        track.append(pos.copy())

    zs = asarray(zs)


    xs, Ps, Pxz, pM, pP = f.batch_filter(zs)
    ms, _, _ = f.rts_smoother(xs, Ps)

    track = asarray(track)
    time = np.arange(0,len(xs)*dt, dt)

    plt.figure()
    plt.subplot(411)
    plt.plot(time, track[:,0])
    plt.plot(time, xs[:,0])
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylabel('x position (m)')
    plt.tight_layout()



    plt.subplot(412)
    plt.plot(time, track[:,1])
    plt.plot(time, xs[:,2])
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylabel('y position (m)')
    plt.tight_layout()


    plt.subplot(413)
    plt.plot(time, xs[:,1])
    plt.plot(time, ms[:,1])
    plt.legend(loc=4)
    plt.ylim([0, 0.2])
    plt.xlabel('time (sec)')
    plt.ylabel('x velocity (m/s)')
    plt.tight_layout()

    plt.subplot(414)
    plt.plot(time, xs[:,3])
    plt.plot(time, ms[:,3])
    plt.ylabel('y velocity (m/s)')
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    DO_PLOT = True
    test_linear_2d_merwe()
    test_linear_2d_merwe_column()
    #test_sigma_plot()
    # test_linear_1d()
    # test_batch_missing_data()
    #
    # test_linear_2d()
    # test_julier_sigma_points_1D()
    #test_simplex_sigma_points_1D()
    # test_fixed_lag()
    # DO_PLOT = True
    # test_rts()
    # kf_circle()
    # test_circle()


    '''test_1D_sigma_points()
    #plot_sigma_test ()

    x = np.array([[1,2]])
    P = np.array([[2, 1.2],
                  [1.2, 2]])\


    kappa = .1

    xi,w = sigma_points (x,P,kappa)
    xm, cov = unscented_transform(xi, w)'''
    #test_radar()
    # test_sigma_plot()
    # test_julier_weights()
    # test_scaled_weights()
    # test_simplex_weights()
    #print('xi=\n',Xi)
    """
    xm, cov = unscented_transform(Xi, W)
    print(xm)
    print(cov)"""
#    sigma_points ([5,2],9*np.eye(2), 2)
    #plt.legend()
    #plt.show()

