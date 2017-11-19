Saver
=====

This is a helper class designed to allow you to save
the state of the Kalman filter for each epoch. Each
instance variable is stored in a list when you call save().

**Example**

.. code::

    saver = Saver(kf)
    for i in range(N):
        kf.predict()
        kf.update(zs[i])
        saver.save()
        
    saver.to_array() # convert all to np.array
    
    # plot the 0th element of kf.x over all epoches
    plot(saver.xs[:, 0])
    

-------

.. automodule:: filterpy.kalman

Kalman filter saver

.. autoclass:: Saver
    :members:

    .. automethod:: __init__
 


