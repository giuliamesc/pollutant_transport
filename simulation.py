# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:10:12 2021

@author: giuli
"""
#Package Loading
import numpy as np
import matplotlib.pyplot as plt
import parameters
import time

# Options
r = 1.0
R = 5.0
dt  = parameters.dt
T = 1
sigma = 2
Q = 1

N = 1000000 # number of simulations
count = 0

maxit = int(T/dt)

# Setting seed for reproducibility
np.random.seed(21)

# Velocity functions
def u1(x,y): 
    return 1 + Q*x/(2*np.pi*(np.power(x,2)+np.power(y,2)))

def u2(x,y):
    return Q*y/(2*np.pi*(np.power(x,2)+np.power(y,2)))


def killing_boundary(x,y):
    # This function receives the coordinate of a point and returns true if that coordinate is inside the well
    if(np.sqrt(np.power(x,2)+np.power(y,2)) <= 1) : return True
    else: return False
start = time.time()
for i in range(N): 
    # Starting value
    X0 = parameters.X0
    Y0 = parameters.Y0
    # Preparation of vectors where we store coordinates over all times until stop
    X = [X0]
    Y = [Y0]
    stop = False
    it = 0
    while(stop == False):
        it = it + 1
        # Generating two independent normals 
        Z1 = np.float(np.random.normal(size = 1))
        Z2 = np.float(np.random.normal(size = 1))
        # Computation of the following position
        new_X = X0 + u1(X0,Y0)*dt + sigma * np.sqrt(dt)*Z1
        new_Y = Y0 + u2(X0,Y0)*dt + sigma * np.sqrt(dt)*Z2
        # Storage of the next position
        X.append(new_X)
        Y.append(new_Y)
        # Checking whether the well is hit
        if (killing_boundary(new_X,new_Y)) : 
            stop = True
            count = count + 1
        if (it > maxit) : stop = True
        X0 = new_X
        Y0 = new_Y
end = time.time()
print("Simulation time: ", end - start, "s; in minutes: ",(end-start)/60) 
ax = plt.gca()
circle1 = plt.Circle((0, 0), r, color='darkturquoise', fill = True)  
ax.plot(X,Y,'lightsteelblue')
ax.plot(X[0],Y[0],'bo')
ax.plot(X[-1],Y[-1], 'ro')
ax.add_patch(circle1)
plt.xlim(-parameters.xl,parameters.xl)
plt.ylim(-parameters.yl,parameters.yl)
ax.set_aspect('equal', adjustable='box')

p_estim = count/N
print(p_estim)
