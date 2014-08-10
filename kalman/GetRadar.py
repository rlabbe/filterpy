import random
import math

"""Copyright 2014 Roger R Labbe Jr.

filterpy library.
http://github.com/rlabbe/filterpy

This is licensed under an MIT license. See the readme.MD file
for more information.
"""



def randn(): return random.gauss(0,1)
def GetRadar(dt):
    """ Simulate radar range to object at 1K altidue and moving at 100m/s.
    Adds about 5% measurement noise. Returns slant range to the object.
    Call once for each new measurement at dt time from last call.
    """
    
    if not hasattr (GetRadar, "posp"):
        GetRadar.posp = 0
        
    vel = 100  + 5 * randn()
    alt = 1000 + 10 * randn()
    pos = GetRadar.posp + vel*dt
    
    v = 0 + pos* 0.05*randn()
    range = math.sqrt (pos**2 + alt**2) + v
    GetRadar.posp = pos
    
    return range
    
    
if __name__ == "__main__":
    for i in range (100):
        print GetRadar (0.1)
