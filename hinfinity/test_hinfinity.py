# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 07:43:57 2014

@author: rlabbe
"""


from filterpy.hinfinity import HInfinityFilter
from numpy import array


dt = 0.1
f = HInfinityFilter(2, 1, 0, gamma=.4)

f.F = array([[1., dt], 
             [0., 1.]])
             
f.H = array([[0., 1.]])
f.x = array([[0., 0.]]).T
f.G = array([[dt**2 / 2, dt]]).T

f.P *= 0.01
f.W = array([[0.0003, 0.005],
             [0.0050, 0.100]])/ 1000

f.V_inv = 1/0.01
f.Q *= 0.01


for i in range(5,40):
        
    f.update (5)
    print(f.x.T)
    f.predict()
'''
a = [1 dt; 0 1]; % transition matrix
b = [dt^2/2; dt]; % input matrix
c = [0 1]; % measurement matrix
x = [0; 0]; % initial state vector
y = c * x; % initial measurement


Pinf = 0.01*eye(2);
W = [0.0003 0.0050; 0.0050 0.1000]/1000;
V = 0.01;
Q = [0.01 0; 0 0.01];


'''