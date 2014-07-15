Repository for various optimal and non-optimal filters implemented in Python.

No documentation as yet. This is code I am developing in conjunction with my
book Kalman Filters and Random Signals in Python. 


Basic use:
--
```
import filterpy.kalman as kf
import filterpy.leastsq as lsq

my_filter = kf.KalmanFilter(3,4)
```

