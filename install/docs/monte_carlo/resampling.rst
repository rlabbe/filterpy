resampling
==========

Routines for resampling particles from particle filters based on
their current weights. All these routines take a list of normalized
weights and returns a list of indexes to the weights that should be
chosen for resampling. The caller is responsible for performing the
actual resample.


.. automodule:: filterpy.monte_carlo

-----

.. autofunction:: residual_resample

-----

.. autofunction:: stratified_resample

-----

.. autofunction:: systematic_resample

-----

.. autofunction:: multinomial_resample
