# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-instance-attributes
"""
Created on Mon Aug  6 07:53:34 2018

@author: rlabbe
"""
from __future__ import (absolute_import, division)
import numpy as np
from numpy import dot, asarray, zeros, outer
from filterpy.common import pretty_str


class IMMEstimator(object):
    """ Implements an Interacting Multiple-Model (IMM) estimator.

    Parameters
    ----------

    filters : (N,) array_like of KalmanFilter objects
        List of N filters. filters[i] is the ith Kalman filter in the
        IMM estimator.

        Each filter must have the same dimension for the state `x` and `P`,
        otherwise the states of each filter cannot be mixed with each other.

    mu : (N,) array_like of float
        mode probability: mu[i] is the probability that
        filter i is the correct one.

    M : (N, N) ndarray of float
        Markov chain transition matrix. M[i,j] is the probability of
        switching from filter j to filter i.


    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.

    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.

    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    N : int
        number of filters in the filter bank

    mu : (N,) ndarray of float
        mode probability: mu[i] is the probability that
        filter i is the correct one.

    M : (N, N) ndarray of float
        Markov chain transition matrix. M[i,j] is the probability of
        switching from filter j to filter i.

    cbar : (N,) ndarray of float
        Total probability, after interaction, that the target is in state j.
        We use it as the # normalization constant.

    likelihood: (N,) ndarray of float
        Likelihood of each individual filter's last measurement.

    omega : (N, N) ndarray of float
        Mixing probabilitity - omega[i, j] is the probabilility of mixing
        the state of filter i into filter j. Perhaps more understandably,
        it weights the states of each filter by:
            x_j = sum(omega[i,j] * x_i)

        with a similar weighting for P_j


    Examples
    --------

    >>> import numpy as np
    >>> from filterpy.common import kinematic_kf
    >>> kf1 = kinematic_kf(2, 2)
    >>> kf2 = kinematic_kf(2, 2)
    >>> # do some settings of x, R, P etc. here, I'll just use the defaults
    >>> kf2.Q *= 0   # no prediction error in second filter
    >>>
    >>> filters = [kf1, kf2]
    >>> mu = [0.5, 0.5]  # each filter is equally likely at the start
    >>> trans = np.array([[0.97, 0.03], [0.03, 0.97]])
    >>> imm = IMMEstimator(filters, mu, trans)
    >>>
    >>> for i in range(100):
    >>>     # make some noisy data
    >>>     x = i + np.random.randn()*np.sqrt(kf1.R[0, 0])
    >>>     y = i + np.random.randn()*np.sqrt(kf1.R[1, 1])
    >>>     z = np.array([[x], [y]])
    >>>
    >>>     # perform predict/update cycle
    >>>     imm.predict()
    >>>     imm.update(z)
    >>>     print(imm.x.T)

    For a full explanation and more examples see my book
    Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


    References
    ----------

    Bar-Shalom, Y., Li, X-R., and Kirubarajan, T. "Estimation with
    Application to Tracking and Navigation". Wiley-Interscience, 2001.

    Crassidis, J and Junkins, J. "Optimal Estimation of
    Dynamic Systems". CRC Press, second edition. 2012.

    Labbe, R. "Kalman and Bayesian Filters in Python".
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, filters, mu, M):
        if len(filters) < 2:
            raise ValueError('filters must contain at least two filters')

        self.filters = filters
        self.mu = asarray(mu) / np.sum(mu)
        self.M = M

        x_shape = filters[0].x.shape
        for f in filters:
            if x_shape != f.x.shape:
                raise ValueError(
                    'All filters must have the same state dimension')

        self.x = zeros(filters[0].x.shape)
        self.P = zeros(filters[0].P.shape)
        self.N = len(filters)  # number of filters
        self.likelihood = zeros(self.N)
        self.omega = zeros((self.N, self.N))
        self._compute_mixing_probabilities()

        # initialize imm state estimate based on current filters
        self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update(self, z):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.
        """

        # run update on each filter, and save the likelihood
        for i, f in enumerate(self.filters):
            f.update(z)
            self.likelihood[i] = f.likelihood

        # update mode probabilities from total probability * likelihood
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        self._compute_mixing_probabilities()

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, u=None):
        """
        Predict next state (prior) using the IMM state propagation
        equations.

        Parameters
        ----------

        u : np.array, optional
            Control vector. If not `None`, it is multiplied by B
            to create the control input into the system.
        """

        # compute mixed initial conditions
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            x = zeros(self.x.shape)
            for kf, wj in zip(self.filters, w):
                x += kf.x * wj
            xs.append(x)

            P = zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                P += wj * (outer(y, y) + kf.P)
            Ps.append(P)

        #  compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            f.x = xs[i].copy()
            f.P = Ps[i].copy()
            f.predict(u)

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def _compute_state_estimate(self):
        """
        Computes the IMM's mixed state estimate from each filter using
        the the mode probability self.mu to weight the estimates.
        """
        self.x.fill(0)
        for f, mu in zip(self.filters, self.mu):
            self.x += f.x * mu

        self.P.fill(0)
        for f, mu in zip(self.filters, self.mu):
            y = f.x - self.x
            self.P += mu * (outer(y, y) + f.P)

    def _compute_mixing_probabilities(self):
        """
        Compute the mixing probability for each filter.
        """

        self.cbar = dot(self.mu, self.M)
        for i in range(self.N):
            for j in range(self.N):
                self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.cbar[j]

    def __repr__(self):
        return '\n'.join([
            'IMMEstimator object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('x_post', self.x_post),
            pretty_str('P_post', self.P_post),
            pretty_str('N', self.N),
            pretty_str('mu', self.mu),
            pretty_str('M', self.M),
            pretty_str('cbar', self.cbar),
            pretty_str('likelihood', self.likelihood),
            pretty_str('omega', self.omega)
            ])
