# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 09:12:44 2021

@author: giuli
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# Entrace in the circular crown with internal radius r and external radius R
def entrance(x,y,r,R):
    # This function receives the coordinate of a point and returns true if that coordinate is inside the well
    radius = np.sqrt(np.power(x,2)+np.power(y,2))
    if(radius <= R and radius > r) : return True
    else: return False
    
# Parameters
X0 = 7.0
Y0 = 7.0
dt = 1e-2 # temporal step
T = 1
sigma = 2
Q = 1
K = int(T/dt)
N = 100 

# Set seed
np.random.seed(21)

# Velocity functions
def u1(x,y): 
    return 1 + Q*x/(2*np.pi*(np.power(x,2)+np.power(y,2)))

def u2(x,y):
    return Q*y/(2*np.pi*(np.power(x,2)+np.power(y,2)))


R = np.arange(8.5,0.5,-1.5) # vector of external radius of my circular crowns
r = R-1.5 
r[-1] = 0 # vector of internal radius of my circular crowns
P = np.zeros(len(R)) # vector to store the probabilities P(tau_j <= T | tau_(j-1) <= T)
iters = 100*np.ones(len(R)) # number of subpaths studied inside each circular crown

def rw(K,X0,Y0,r,R):
    
    finished_mc=0
    x_mc = X0
    y_mc = Y0
    xs = []
    ys = []
    xs.append(X0)
    ys.append(Y0)
    for i in range(K):   
        Z1 = np.float(np.random.normal(size = 1))
        Z2 = np.float(np.random.normal(size = 1))       
        # Moves crude MC
        u1_old = u1(x_mc,y_mc)
        u2_old = u2(x_mc,y_mc)
        x_mc = x_mc + u1_old*dt + sigma * np.sqrt(dt)*Z1
        y_mc = y_mc + u2_old*dt + sigma * np.sqrt(dt)*Z2
        xs.append(x_mc)
        ys.append(y_mc)
        if (entrance(x_mc,y_mc,r,R)) : 
            finished_mc=1
           
    return finished_mc,xs,ys

# ax = plt.gca()
# circle1 = plt.Circle((0, 0), R[-1], color='darkturquoise', fill = True)
# ax.add_patch(circle1)
# for i in range(len(R)-1):
#     circle = plt.Circle((0, 0), R[i], color='green', fill = False)
#     ax.add_patch(circle)
# plt.xlim(-9,9)
# plt.ylim(-9,9)
# ax.set_aspect('equal', adjustable='box')
# ax.plot(X0,Y0, 'ro')

X0s_old = [X0]
Y0s_old = [Y0]

start = time.time()

for j in range(len(R)): # for each level
    K_r = int(iters[j])
    X0s_new = []
    Y0s_new = []
    for n in range(len(X0s_old)): # for each initial point of the previous state
        for k in range(K_r): # K_r subpaths
            # Storage of the subpath
            xs = []
            ys = []
            hit,xs,ys = rw(K,X0s_old[n],Y0s_old[n],r[j],R[j])
            #ax.plot(xs,ys)
            if(hit == 1):
                #ax.plot(xs[-1],ys[-1],'bo')
                X0s_new.append(xs[-1]) # store new intial point for next state
                Y0s_new.append(ys[-1])
                P[j] = P[j] + 1
    P[j] = P[j] / (K_r * len(X0s_old))
    X0s_old = X0s_new
    Y0s_old = Y0s_new
    
end = time.time()
print("Simulation time: ", end - start, "s; in minutes: ",(end-start)/60) 
res = np.prod(P)
print(res)
