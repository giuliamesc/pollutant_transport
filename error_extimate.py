# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:37:41 2022

@author: giuli
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:10:12 2021

@author: giuli
"""
#Package Loading
import numpy as np
import scipy
import matplotlib.pyplot as plt
import parameters
import time

# Options
r = 1.0
R = 5.0
dt  = (1e-2)/2
T = 1
sigma = 2
Q = 1
alpha = 0.05
C_alpha=scipy.stats.norm.ppf(1-alpha/2)

N = 10000 # set N

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
    if(np.sqrt(np.power(x,2)+np.power(y,2)) <= r) : return True
    else: return False
    
def path(N):
    count = 0
    Z_mc = np.zeros([N,])
    for i in range(N): 
        # Starting value
        X0 = parameters.X0
        Y0 = parameters.Y0
        # Preparation of vectors where we store coordinates over all times until stop
        X_fine = [X0]
        Y_fine = [Y0]
        X_coarse = [X0]
        Y_coarse = [X0]
        stop = False
        it = 0
        while(stop == False):
            # Generating two independent normals 
            Z1 = np.float(np.random.normal(size = 1))
            Z2 = np.float(np.random.normal(size = 1))
            # Computation of the following position
            new_X_fine = X0 + u1(X0,Y0)*dt + sigma * np.sqrt(dt)*Z1
            new_Y_fine = Y0 + u2(X0,Y0)*dt + sigma * np.sqrt(dt)*Z2
            # Storage of the next position
            X_fine.append(new_X_fine)
            Y_fine.append(new_Y_fine)
            if np.mod(it,2)==1 :
                X_coarse.append(new_X_fine)
                Y_coarse.append(new_Y_fine)
            # Checking wheter T is reached
            if (it > maxit) : stop = True
            # Checking whether the well is hit
            if (killing_boundary(new_X_fine,new_Y_fine)) : 
                stop = True
                count = count + 1
                Z_mc[i] = 1
            it = it + 1
            X0 = new_X_fine
            Y0 = new_Y_fine
    return count/N, X_fine, Y_fine, X_coarse, Y_coarse, Z_mc
    
start = time.time()

mu, X_fine, Y_fine, X_coarse, Y_coarse, Z_mc = path(N)

sigma = np.var(Z_mc)

print('Variance: ', sigma)
print('Extimated probability: ', mu)

print('Confidence Interval: [', mu, '+-', C_alpha * np.sqrt(sigma)/np.sqrt(N),']')


end = time.time()

print("Simulation time: ", end - start, "s; in minutes: ",(end-start)/60) 

# Plots
ax1 = plt.gca()
circle1 = plt.Circle((0, 0), r, color='darkturquoise', fill = True)  
circle2 = plt.Circle((0, 0), 2, color='darkturquoise', fill = False) 
circle3 = plt.Circle((0, 0), 3, color='darkturquoise', fill = False) 
circle4 = plt.Circle((0, 0), 4, color='darkturquoise', fill = False) 
circle5 = plt.Circle((0, 0), 5, color='darkturquoise', fill = False) 
ax1.plot(X_fine,Y_fine,'forestgreen', linewidth = 0.5)
ax1.plot(X_fine[0],Y_fine[0],'bo')
ax1.plot(X_fine[-1],Y_fine[-1], 'ro')
ax1.add_patch(circle1)
ax1.add_patch(circle2)
ax1.add_patch(circle3)
ax1.add_patch(circle4)
ax1.add_patch(circle5)
plt.xlim(-parameters.xl,parameters.xl)
plt.ylim(-parameters.yl,parameters.yl)
ax1.set_aspect('equal', adjustable='box')


ax2 = plt.gca()
circle1 = plt.Circle((0, 0), r, color='darkturquoise', fill = True)  
circle2 = plt.Circle((0, 0), 2, color='darkturquoise', fill = False) 
circle3 = plt.Circle((0, 0), 3, color='darkturquoise', fill = False) 
circle4 = plt.Circle((0, 0), 4, color='darkturquoise', fill = False) 
circle5 = plt.Circle((0, 0), 5, color='darkturquoise', fill = False) 
ax2.plot(X_coarse,Y_coarse,'blue', linewidth = 0.5)
ax2.plot(X_fine[0],Y_fine[0],'bo')
ax2.plot(X_fine[-1],Y_fine[-1], 'ro')
ax2.add_patch(circle1)
ax2.add_patch(circle2)
ax2.add_patch(circle3)
ax2.add_patch(circle4)
ax2.add_patch(circle5)
plt.xlim(-parameters.xl,parameters.xl)
plt.ylim(-parameters.yl,parameters.yl)
ax2.set_aspect('equal', adjustable='box')

X_coarse_new = np.zeros(len(X_fine))
Y_coarse_new = np.zeros(len(Y_fine))

for i in range(len(X_fine)):
    if np.mod(i,2)==0:
        X_coarse_new[i] = X_fine[i]
        Y_coarse_new[i] = Y_fine[i]
    else:
        X_coarse_new[i] = 0.5 * (X_fine[i-1] + X_fine[i+1])
        Y_coarse_new[i] = 0.5 * (Y_fine[i-1] + Y_fine[i+1])
        
err_inf_X = np.max(np.abs(X_coarse_new-X_fine))
err_inf_Y = np.max(np.abs(Y_coarse_new-Y_fine))

print('Error on x: ', err_inf_X)
print('Error on y: ', err_inf_Y)

rad = np.sqrt(parameters.X0**2+parameters.Y0**2)
print('Normalized error on x: ', err_inf_X/rad)
print('Normalized error on y: ', err_inf_Y/rad)

