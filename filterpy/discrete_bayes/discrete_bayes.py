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
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage.interpolation import shift


def normalize(pdf):
    """Normalize distribution `pdf` in-place so it sums to 1.0.

    Returns pdf for convienence, so you can write things like:

    >>> kernel = normalize(randn(7))

    Parameters
    ----------

    pdf : ndarray
        discrete distribution that needs to be converted to a pdf. Converted
        in-place, i.e., this is modified.

    Returns
    -------

    pdf : ndarray
        The converted pdf.
    """

    pdf /= sum(np.asarray(pdf, dtype=float))
    return pdf


def update(likelihood, prior):
    """ Computes the posterior of a discrete random variable given a
    discrete likelihood and prior. In a typical application the likelihood
    will be the likelihood of a measurement matching your current environment,
    and the prior comes from discrete_bayes.predict().

    Parameters
    ----------

    likelihood : ndarray, dtype=flaot
         array of likelihood values

    prior : ndarray, dtype=flaot
        prior pdf.

    Returns
    -------

    posterior : ndarray, dtype=float
        Returns array representing the posterior.


    Examples
    --------
    .. code-block:: Python

        # self driving car. Sensor returns values that can be equated to positions
        # on the road. A real likelihood compuation would be much more complicated
        # than this example.

        likelihood = np.ones(len(road))
        likelihood[road==z] *= scale_factor

        prior = predict(posterior, velocity, kernel)
        posterior = update(likelihood, prior)
    """

    posterior = prior * likelihood
    return normalize(posterior)



def predict(pdf, offset, kernel, mode='wrap', cval=0.):
    """ Performs the discrete Bayes filter prediction step, generating
    the prior.

    `pdf` is a discrete probability distribution expressing our initial
    belief.

    `offset` is an integer specifying how much we want to move to the right
    (negative values means move to the left)

    We assume there is some noise in that offset, which we express in `kernel`.
    For example, if offset=3 and kernel=[.1, .7., .2], that means we think
    there is a 70% chance of moving right by 3, a 10% chance of moving 2
    spaces, and a 20% chance of moving by 4.

    It returns the resulting distribution.

    If `mode='wrap'`, then the probability distribution is wrapped around
    the array.

    If `mode='constant'`, or any other value the pdf is shifted, with `cval`
    used to fill in missing elements.

    Examples
    --------
    .. code-block:: Python

        belief = [.05, .05, .05, .05, .55, .05, .05, .05, .05, .05]
        prior = predict(belief, offset=2, kernel=[.1, .8, .1])
    """

    if mode == 'wrap':
        return convolve(np.roll(pdf, offset), kernel, mode='wrap')

    return convolve(shift(pdf, offset, cval=cval), kernel,
                    cval=cval, mode='constant')
