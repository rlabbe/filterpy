# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 09:22:00 2014

@author: rlabbe
"""
from __future__ import division
import matplotlib.pyplot as plt

h1s = []
h2s = []

for n in range (100):
    
    g = (2*(2*n + 1))/ ((n+2)*(n+1))
    
    h1 = 6 / ((n+2)*(n+1))
    
    h2 = 4 - 2*g - (4*(g-2)**2 - 3*g**2)**.5
    
    print h1-h2
    
    h1s.append (h1)
    h2s.append(h2)
    
#plt.plot(h1s)
#plt.plot(h2s)
    
for i in range(10):
    plt.scatter (i,10)

    
    