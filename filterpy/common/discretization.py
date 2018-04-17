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


from numpy import array, zeros, vstack, eye
from scipy.linalg import expm, inv, block_diag


def Q_discrete_white_noise(dim, dt=1., var=1., block_size=1):
    """
    Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
    is the variance in the noise.

    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.

    Parameters
    -----------

    dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)

    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    var : float, default=1.0
        variance in the noise

    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.


    Examples
    --------
    >>> # constant velocity model in a 3D world with a 10 Hz update rate
    >>> Q_discrete_white_noise(2, dt=0.1, var=1., block_size=3)
    array([[0.000025, 0.0005  , 0.      , 0.      , 0.      , 0.      ],
           [0.0005  , 0.01    , 0.      , 0.      , 0.      , 0.      ],
           [0.      , 0.      , 0.000025, 0.0005  , 0.      , 0.      ],
           [0.      , 0.      , 0.0005  , 0.01    , 0.      , 0.      ],
           [0.      , 0.      , 0.      , 0.      , 0.000025, 0.0005  ],
           [0.      , 0.      , 0.      , 0.      , 0.0005  , 0.01    ]])

    References
    ----------

    Bar-Shalom. "Estimation with Applications To Tracking and Navigation".
    John Wiley & Sons, 2001. Page 274.
    """

    assert dim == 2 or dim == 3 or dim == 4
    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [ .5*dt**3,    dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [ .5*dt**3,    dt**2,       dt],
             [ .5*dt**2,       dt,        1]]
    else:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
             [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
             [(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]

    return block_diag(*[Q]*block_size) * var


def Q_continuous_white_noise(dim, dt=1., spectral_density=1.,
                             block_size=1):
    """
    Returns the Q matrix for the Discretized Continuous White Noise
    Model. dim may be either 2, 3, 4, dt is the time step, and sigma is the
    variance in the noise.

    Parameters
    ----------

    dim : int (2 or 3 or 4)
        dimension for Q, where the final dimension is (dim x dim)
        2 is constant velocity, 3 is constant acceleration, 4 is
        constant jerk

    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    spectral_density : float, default=1.0
        spectral density for the continuous process

    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.

    Examples
    --------

    >>> # constant velocity model in a 3D world with a 10 Hz update rate
    >>> Q_continuous_white_noise(2, dt=0.1, block_size=3)
    array([[0.00033333, 0.005     , 0.        , 0.        , 0.        , 0.        ],
           [0.005     , 0.1       , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.00033333, 0.005     , 0.        , 0.        ],
           [0.        , 0.        , 0.005     , 0.1       , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.00033333, 0.005     ],
           [0.        , 0.        , 0.        , 0.        , 0.005     , 0.1       ]])
    """

    assert dim == 2 or dim == 3 or dim == 4
    if dim == 2:
        Q = [[(dt**3)/3., (dt**2)/2.],
             [(dt**2)/2.,    dt]]
    elif dim == 3:
        Q = [[(dt**5)/20., (dt**4)/8., (dt**3)/6.],
             [ (dt**4)/8., (dt**3)/3., (dt**2)/2.],
             [ (dt**3)/6., (dt**2)/2.,        dt]]

    else:
        Q = [[(dt**7)/252., (dt**6)/72., (dt**5)/30., (dt**4)/24.],
             [(dt**6)/72.,  (dt**5)/20., (dt**4)/8.,  (dt**3)/6.],
             [(dt**5)/30.,  (dt**4)/8.,  (dt**3)/3.,  (dt**2)/2.],
             [(dt**4)/24.,  (dt**3)/6.,  (dt**2/2.),   dt]]

    return block_diag(*[Q]*block_size) * spectral_density


def van_loan_discretization(F, G, dt):

    """ Discretizes a linear differential equation which includes white noise
    according to the method of C. F. van Loan [1]. Given the continuous
    model

        x' =  Fx + Gu

    where u is the unity white noise, we compute and return the sigma and Q_k
    that discretizes that equation.


    Examples
    --------

        Given y'' + y = 2u(t), we create the continuous state model of

        x' = [ 0 1] * x + [0]*u(t)
             [-1 0]       [2]

        and a time step of 0.1:


        >>> F = np.array([[0,1],[-1,0]], dtype=float)
        >>> G = np.array([[0.],[2.]])
        >>> phi, Q = van_loan_discretization(F, G, 0.1)

        >>> phi
        array([[ 0.99500417,  0.09983342],
               [-0.09983342,  0.99500417]])

        >>> Q
        array([[ 0.00133067,  0.01993342],
               [ 0.01993342,  0.39866933]])

        (example taken from Brown[2])


    References
    ----------

    [1] C. F. van Loan. "Computing Integrals Involving the Matrix Exponential."
        IEEE Trans. Automomatic Control, AC-23 (3): 395-404 (June 1978)

    [2] Robert Grover Brown. "Introduction to Random Signals and Applied
        Kalman Filtering." Forth edition. John Wiley & Sons. p. 126-7. (2012)
    """


    n = F.shape[0]

    A = zeros((2*n, 2*n))

    # we assume u(t) is unity, and require that G incorporate the scaling term
    # for the noise. Hence W = 1, and GWG' reduces to GG"

    A[0:n,     0:n] = -F.dot(dt)
    A[0:n,   n:2*n] = G.dot(G.T).dot(dt)
    A[n:2*n, n:2*n] = F.T.dot(dt)

    B=expm(A)

    sigma = B[n:2*n, n:2*n].T

    Q = sigma.dot(B[0:n, n:2*n])

    return (sigma, Q)


def linear_ode_discretation(F, L=None, Q=None, dt=1):
    n = F.shape[0]

    if L is None:
        L = eye(n)

    if Q is None:
        Q = zeros((n,n))

    A = expm(F*dt)

    phi = zeros((2*n, 2*n))

    phi[0:n,     0:n] = F
    phi[0:n,   n:2*n] = L.dot(Q).dot(L.T)
    phi[n:2*n, n:2*n] = -F.T

    zo = vstack((zeros((n,n)), eye(n)))

    CD = expm(phi*dt).dot(zo)

    C = CD[0:n,:]
    D = CD[n:2*n,:]
    q = C.dot(inv(D))

    return (A, q)







