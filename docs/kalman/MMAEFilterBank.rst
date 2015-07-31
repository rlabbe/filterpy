MMAE Filter Bank
================

needs documentation....


**Example**


.. code::

    from filterpy.kalman import MMAEFilterBank

    pos, zs = generate_data(120, noise_factor=0.2)
    z_xs = zs[:, 0]
    t = np.arange(0, len(z_xs) * dt, dt)

    dt = 0.1
    filters = [make_cv_filter(dt), make_ca_filter(dt)]
    H_cv = np.array([[1., 0, 0],
                     [0., 1, 0]])

    H_ca = np.array([[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])


    bank = MMAEFilterBank(filters, (0.5, 0.5), dim_x=3, H=(H_cv, H_ca))

    xs, probs = [], []
    for z in z_xs:
        bank.predict()
        bank.update(z)
        xs.append(bank.x[0])
        probs.append(bank.p[0])

    plt.subplot(121)
    plt.plot(xs)
    plt.subplot(122)
    plt.plot(probs)


-------

.. automodule:: filterpy.kalman




.. autoclass:: MMAEFilterBank
    :members:    
    
    .. automethod:: __init__

