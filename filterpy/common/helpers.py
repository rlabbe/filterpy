# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, bare-except

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


from __future__ import print_function
from collections import defaultdict, deque
import copy
import inspect
import numpy as np


class Saver(object):
    """
    Helper class to save the states of any filter object.
    Each time you call save() all of the attributes (state, covariances, etc)
    are appended to lists.

    Generally you would do this once per epoch - predict/update.

    Then, you can access any of the states by using the [] syntax or by
    using the . operator.

    .. code-block:: Python

        my_saver = Saver()
        ... do some filtering

        x = my_saver['x']
        x = my_save.x

    Either returns a list of all of the state `x` values for the entire
    filtering process.

    If you want to convert all saved lists into numpy arrays, call to_array().


    Parameters
    ----------

    kf : object
        any object with a __dict__ attribute, but intended to be one of the
        filtering classes

    save_current : bool, default=False
        save the current state of `kf` when the object is created;

    skip_private: bool, default=False
        Control skipping any private attribute (anything starting with '_')
        Turning this on saves memory, but slows down execution a bit.

    skip_callable: bool, default=False
        Control skipping any attribute which is a method. Turning this on
        saves memory, but slows down execution a bit.

    ignore: (str,) tuple of strings
        list of keys to ignore.

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

        x = np.array(s.x) # get the kf.x state in an np.array
        plt.plot(x[:, 0], x[:, 2])

        # ... or ...
        s.to_array()
        plt.plot(s.x[:, 0], s.x[:, 2])

    """

    def __init__(
        self, kf, save_current=False, skip_private=False, skip_callable=False, ignore=()
    ):
        """Construct the save object, optionally saving the current
        state of the filter"""
        # pylint: disable=too-many-arguments

        self._kf = kf
        self._DL = defaultdict(list)
        self._skip_private = skip_private
        self._skip_callable = skip_callable
        self._ignore = ignore
        self._len = 0

        # need to save all properties since it is possible that the property
        # is computed only on access. I use this trick a lot to minimize
        # computing unused information.
        properties = inspect.getmembers(
            type(kf), lambda o: isinstance(o, property)
        )
        self.properties = [p for p in properties if p[0] not in ignore]

        if save_current:
            self.save()

    def save(self):
        """save the current state of the Kalman filter"""

        kf = self._kf

        # force all attributes to be computed. this is only necessary
        # if the class uses properties that compute data only when
        # accessed
        for prop in self.properties:
            self._DL[prop[0]].append(getattr(kf, prop[0]))

        v = copy.deepcopy(kf.__dict__)

        if self._skip_private:
            for key in list(v.keys()):
                if key.startswith("_"):
                    del v[key]

        if self._skip_callable:
            for key in list(v.keys()):
                if callable(v[key]):
                    del v[key]

        for ig in self._ignore:
            if ig in v:
                del v[ig]

        for key in list(v.keys()):
            self._DL[key].append(v[key])

        self.__dict__.update(self._DL)
        self._len += 1

    def __getitem__(self, key):
        return self._DL[key]

    def __setitem__(self, key, newvalue):
        self._DL[key] = newvalue
        self.__dict__.update(self._DL)

    def __len__(self):
        return self._len

    @property
    def keys(self):
        """list of all keys"""
        return list(self._DL.keys())

    def to_array(self, flatten=False):
        """
        Convert all saved attributes from a list to np.array.

        This may or may not work - every saved attribute must have the
        same shape for every instance. i.e., if `K` changes shape due to `z`
        changing shape then the call will raise an exception.

        This can also happen if the default initialization in __init__ gives
        the variable a different shape then it becomes after a predict/update
        cycle.
        """
        for key in self.keys:
            try:
                self.__dict__[key] = np.array(self._DL[key])
            except:
                # get back to lists so we are in a valid state
                self.__dict__.update(self._DL)
                raise ValueError("could not convert {} into np.array".format(key))
        if flatten:
            self.flatten()

    def flatten(self):
        """
        Flattens any np.array of column vectors into 1D arrays. Basically,
        this makes data readable for humans if you are just inspecting via
        the REPL. For example, if you have saved a KalmanFilter object with 89
        epochs, self.x will be shape (89, 9, 1) (for example). After flatten
        is run, self.x.shape == (89, 9), which displays nicely from the REPL.

        There is no way to unflatten, so it's a one way trip.
        """

        for key in self.keys:
            try:
                arr = self.__dict__[key]
                shape = arr.shape
                if shape[2] == 1:
                    self.__dict__[key] = arr.reshape(shape[0], shape[1])
                arr = self.__dict__[key]
                shape = arr.shape
                if len(shape) == 2 and shape[1] == 1:
                    self.__dict__[key] = arr.ravel()
            except:
                # not an ndarray or not a column vector
                pass

    def __repr__(self):
        return "<Saver object at {}\n  Keys: {}>".format(
            hex(id(self)), " ".join(self.keys)
        )


def runge_kutta4(y, x, dx, f):
    """computes 4th order Runge-Kutta for dy/dx.

    Parameters
    ----------

    y : scalar
        Initial/current value for y
    x : scalar
        Initial/current value for x
    dx : scalar
        difference in x (e.g. the time step)
    f : ufunc(y,x)
        Callable function (y, x) that you supply to compute dy/dx for
        the specified values.

    """

    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5 * k1, x + 0.5 * dx)
    k3 = dx * f(y + 0.5 * k2, x + 0.5 * dx)
    k4 = dx * f(y + k3, x + dx)

    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def pretty_str(label, arr):
    """
    Generates a pretty printed NumPy array with an assignment. Optionally
    transposes column vectors so they are drawn on one line. Strictly speaking
    arr can be any time convertible by `str(arr)`, but the output may not
    be what you want if the type of the variable is not a scalar or an
    ndarray.

    Examples
    --------
    >>> pprint('cov', np.array([[4., .1], [.1, 5]]))
    cov = [[4.  0.1]
           [0.1 5. ]]

    >>> print(pretty_str('x', np.array([[1], [2], [3]])))
    x = [[1 2 3]].T
    """

    def is_col(a):
        """return true if a is a column vector"""
        try:
            return a.shape[0] > 1 and a.shape[1] == 1
        except (AttributeError, IndexError):
            return False

    # display empty lists correctly
    try:
        if len(arr) == 0:
            return label + " = " + str(type(arr)())
    except TypeError:
        pass

    if type(arr) is list or type(arr) is tuple or type(arr) is deque:
        return "\n".join(
            [pretty_str(label + "[" + str(i) + "]", x) for (i, x) in enumerate(arr)]
        )

    if label is None:
        label = ""

    if label:
        label += " = "

    if is_col(arr):
        return label + str(arr.T).replace("\n", "") + ".T"

    rows = str(arr).split("\n")
    if not rows:
        return ""

    s = label + rows[0]
    pad = " " * len(label)
    for line in rows[1:]:
        s = s + "\n" + pad + line

    return s


def pprint(label, arr, **kwargs):
    """pretty prints an NumPy array using the function pretty_str. Keyword
    arguments are passed to the print() function.

    See Also
    --------
    pretty_str

    Examples
    --------
    >>> pprint('cov', np.array([[4., .1], [.1, 5]]))
    cov = [[4.  0.1]
           [0.1 5. ]]
    """

    print(pretty_str(label, arr), **kwargs)


def reshape_z(z, dim_z, ndim):
    """ensure z is a (dim_z, 1) shaped vector"""

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError(
            "z (shape {}) must be convertible to shape ({}, 1)".format(z.shape, dim_z)
        )

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z


def inv_diagonal(S):
    """
    Computes the inverse of a diagonal NxN np.array S. In general this will
    be much faster than calling np.linalg.inv().

    However, does NOT check if the off diagonal elements are non-zero. So long
    as S is truly diagonal, the output is identical to np.linalg.inv().

    Parameters
    ----------
    S : np.array
        diagonal NxN array to take inverse of

    Returns
    -------
    S_inv : np.array
        inverse of S


    Examples
    --------

    This is meant to be used as a replacement inverse function for
    the KalmanFilter class when you know the system covariance S is
    diagonal. It just makes the filter run faster, there is

    >>> kf = KalmanFilter(dim_x=3, dim_z=1)
    >>> kf.inv = inv_diagonal  # S is 1x1, so safely diagonal
    """

    S = np.asarray(S)

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square Matrix")

    si = np.zeros(S.shape)
    for i in range(len(S)):
        si[i, i] = 1.0 / S[i, i]
    return si


def outer_product_sum(A, B=None):
    r"""
    Computes the sum of the outer products of the rows in A and B

        P = \Sum {A[i] B[i].T} for i in 0..N

        Notionally:

        P = 0
        for y in A:
            P += np.outer(y, y)

    This is a standard computation for sigma points used in the UKF, ensemble
    Kalman filter, etc., where A would be the residual of the sigma points
    and the filter's state or measurement.

    The computation is vectorized, so it is much faster than the for loop
    for large A.

    Parameters
    ----------
    A : np.array, shape (M, N)
        rows of N-vectors to have the outer product summed

    B : np.array, shape (M, N)
        rows of N-vectors to have the outer product summed
        If it is `None`, it is set to A.

    Returns
    -------
    P : np.array, shape(N, N)
        sum of the outer product of the rows of A and B

    Examples
    --------

    Here sigmas is of shape (M, N), and x is of shape (N). The two sets of
    code compute the same thing.

    >>> P = outer_product_sum(sigmas - x)
    >>>
    >>> P = 0
    >>> for s in sigmas:
    >>>     y = s - x
    >>>     P += np.outer(y, y)
    """

    if B is None:
        B = A

    outer = np.einsum("ij,ik->ijk", A, B)
    return np.sum(outer, axis=0)


def compare_kf(kf1, kf2, log=True, **kwargs):
    """Compare two Kalman filters.

    For each variable each object has in common (x, P, S, K, etc) compare
    them using np.allclose().

    Prints a report if `log` is true, and returns a list of names if any
    are different, otherwise prints nothing and returns None.
    """

    # get variables common to both objects
    v1, v2 = vars(kf1), vars(kf2)
    k1, k2 = set(v1.keys()), set(v2.keys())
    attrs = k2.intersection(k1)

    different_keys = []
    for attr in attrs:
        if attr[0] == "_":
            continue

        if not np.allclose(v1[attr], v2[attr], **kwargs):
            if log:
                print(attr, "is different")
                print(pretty_str(attr, v1[attr]))
                print(pretty_str(attr, v2[attr]))
                print()
            different_keys.append(attr)

    if len(different_keys) > 0:
        return different
    else:
        return None


def copy_states(dst, src):
    """Copy filter states from `src` to `dst`.

    for each variable that `dst` and `src` have in common, use
    np.copy() to copy from the source to the destination.

    This has the potential of breaking things if you are using two different
    types - the destination could end up in an incorrect state since not
    all variables may be initalized correctly.

    The main use case is for testing or comparing different algorithms

        kf1 = KalmanFilter()
        kf1.F = ...
        kf1.P = ...
        kf2 = KalmanFilter()
        # reuse F, P, etc from kf1
        copy_states(kf2, kf1)

        for z in obs:
            kf1.predict()
            kf2.predict()
            ...
    """
    # get variables common to both objects
    v1, v2 = vars(dst), vars(src)
    k1, k2 = set(v1.keys()), set(v2.keys())
    attrs = k2.intersection(k1)

    for key in attrs:
        val = getattr(src, key)
        if type(val).__name__ not in ["method", "function"]:
            dst.__dict__[key] = copy.deepcopy(val)


def repr_string(obj, private=True):
    """Generate a __repr_ string for an filter object.

    It will pretty print numpy arrays to be readable, and display lists
    with indexed values. It also gathers up all properties.
    """

    keys = obj.__dir__()
    keys = [key for key in keys if key[:2] != "__"]
    if not private:
        keys = [key for key in keys if key[0] != "_"]

    s = []
    for key in keys:
        val = getattr(obj, key)
        if type(val).__name__ not in ["method", "function"]:
            s.append(pretty_str(key, val))
    return type(obj).__name__ + " object\n" + "\n".join(s)
