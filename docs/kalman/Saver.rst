Saver
=====

This is a helper class designed to allow you to save
the state of the Kalman filter for each epoch. Each
instance variable is stored in a list when you call save().

This class is deprecated as of version 1.3.2 and will 
be deleted soon. Instead, see the class 
filterpy.common.Saver, which works for any class, not
just a KalmanFilter object.



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
 


