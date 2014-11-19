# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:03:38 2014

@author: rlabbe
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, FixedLagSmoother, rts_smoother


fls = FixedLagSmoother(dim_x=2, dim_z=1)

fls.x = np.array([[0.],
                  [.5]])

fls.F = np.array([[1.,1.],
                  [0.,1.]])

fls.H = np.array([[1.,0.]])

fls.P *= 200                 
fls.R *= 5.                     
fls.Q *= 0.001


kf = KalmanFilter(dim_x=2, dim_z=1)

kf.x = np.array([[0.],
                 [.5]])

kf.F = np.array([[1.,1.],
                 [0.,1.]])

kf.H = np.array([[1.,0.]])
   
kf.P *= 2000              
kf.R *= 1.                     
kf.Q *= 0.001


N = 4 # size of lag

nom =  np.array([t/2. for t in range (0,40)])
zs = np.array([t + random.randn()*1.1 for t in nom])

xs, x = fls.smooth_batch(zs, N)

M,P,_,_ = kf.batch_filter(zs)
rks_x,_,_ = rts_smoother(M, P, kf.F, kf.Q)

xfl = xs[:,0].T[0]
xkf = M[:,0].T[0]

plt.cla()
plt.plot(zs,'o', alpha=0.5, marker='o', label='zs')
plt.plot(x[:,0], label='FLS')
plt.plot(xfl, label='FLS S')
plt.plot(xkf, label='KF')
plt.plot(rks_x[:,0], label='RKS')
plt.legend(loc=4)
plt.show()
 



fl_res = abs(xfl-nom)
kf_res = abs(xkf-nom)
print(fl_res)
print(kf_res)

print('std fixed lag:', np.mean(fl_res[N:]))
print('std kalman:', np.mean(kf_res[N:]))

'''
for i in range(N, len(zs)):
    x = fk.smooth(zs[i-N+1:i+1])
    print(x)








mu, cov, _, _ = fk.batch_filter (zs)
mus = [x[0,0] for x in mu]

M,P,C = rts_smoother(mu, cov, fk.F, fk.Q)



# plot data
p1, = plt.plot(zs,'cyan', alpha=0.5)
p2, = plt.plot (M[:,0],c='b')
p3, = plt.plot (mus,c='r')
p4, = plt.plot ([0,len(zs)],[0,len(zs)], 'g') # perfect result
plt.legend([p1,p2, p3, p4],
           ["measurement", "RKS", "KF output", "ideal"], 4)


plt.show()
'''