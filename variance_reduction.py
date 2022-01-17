# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 17:35:26 2021

@author: giuli
"""

"""
This file contains the code to solve the second question of the assignment.
The proposed solution for Variance Reduction relies on Antithetic Variables. 
"""

# Package Loading
import numpy as np
import scipy.stats as st
import parameters
import matplotlib.pyplot as plt


# Options and parameters
N = 31430 # number of samples: same as in the previous point 
N2 = int(N/2) # number of samples for AV
dt = 1e-2 # time discretization step
T = 1 # time limit
Q = 1
sigma = 2 # diffusion factor
r = 1.0 # radius of the well
K = int(T/dt) # iteration corresponding to the desired stopping time
alpha = 0.05 # confidence level 
C_alpha=st.norm.ppf(1-alpha/2) # corresponding quantile of the standard normal

# Setting seed for reproducibility
np.random.seed(21)

# Vectors for storage
Z_mc=np.zeros(N)
Z_mc2=np.zeros(N2)
Z_av=np.zeros(N2)

# Velocity functions
def u1(x,y): 
    return 1 + Q*x/(2*np.pi*(np.power(x,2)+np.power(y,2)))

def u2(x,y):
    return Q*y/(2*np.pi*(np.power(x,2)+np.power(y,2)))

# Killing boundary
def killing_boundary(x,y):
    # This function receives the coordinate of a point and returns true if that coordinate is inside the well
    if(np.sqrt(np.power(x,2)+np.power(y,2)) <= r) : return True
    else: return False

# Defining the trajectories for CMC and Antithetic Variables
def paths(K):
    # To store whether the trajectory hits the well
    finished_mc = 0
    finished_av = 0
    # Initial Positions
    x_mc = parameters.X0
    y_mc = parameters.Y0
    x_av = parameters.X0
    y_av = parameters.Y0
    # Path Storage
    X = [x_mc]
    Y = [y_mc]
    Xav = [x_av]
    Yav = [y_av]
    
    for i in range(K):   
        
        # Generating two independent standard normals
        Z1 = np.float(np.random.normal(size = 1))
        Z2 = np.float(np.random.normal(size = 1))
        
        # Storing stop values in order to stop updates when the well is hit
        stop_mc = False 
        stop_av = False
        
        # Updating crude MC
        if stop_mc == False:
            # Computation of the velocities (cannot be done after on the fly, since we do not store the previous position practically - just in the vector)
            u1_old = u1(x_mc,y_mc)
            u2_old = u2(x_mc,y_mc)
            # Position update
            x_mc = x_mc + u1_old*dt + sigma * np.sqrt(dt)*Z1
            y_mc = y_mc + u2_old*dt + sigma * np.sqrt(dt)*Z2
            # Position storage
            X.append(x_mc)
            Y.append(y_mc)
            # Checking wheter the path hits the well
            if (killing_boundary(x_mc,y_mc)) : 
                finished_mc = 1
                stop_mc = True
                
        # Updating AV with the same random realization 
        if stop_av == False:   
            # Computation of the velocities (cannot be done after on the fly, since we do not store the previous position practically - just in the vector)
            u1_old = u1(x_av,y_av)
            u2_old = u2(x_av,y_av)
            # Position update with minus (antithetic path)
            x_av = x_av + u1_old*dt - sigma * np.sqrt(dt)*Z1
            y_av = y_av + u2_old*dt - sigma * np.sqrt(dt)*Z2
            # Position storage
            Xav.append(x_av)
            Yav.append(y_av)
            # Checking wheter the path hits the well
            if (killing_boundary(x_av,y_av)) : 
                finished_av = 1
                stop_av = True
           
    return finished_mc,finished_av,X,Y,Xav,Yav

# Crude Monte Carlo
for i in range(int(N)):    
    Z_mc[i],_,_,_,_,_ = paths(K)

mean_cmc=np.mean(Z_mc) # esitmate of CMC
var_cmc=np.var(Z_mc, ddof = 1) # variance of CMC
print('Variance CMC: ', var_cmc)
ci_cmc=C_alpha*np.sqrt(var_cmc)/np.sqrt(N) # semilength of confidence interval

# Antithetic Variables
for i in range(int(N2)):    
    Z_mc2[i],Z_av[i],X,Y,Xav,Yav = paths(K)
    
Y_av = 0.5*(Z_mc2+Z_av) # variance of AV
mean_av=np.mean(Y_av) # estimate of AV
var_av = np.var(Y_av, ddof = 1)
print('Variance AV: ', var_av)
C=np.cov(Z_mc2,Z_av) # covariance between Z and Z_av -- NEGATIVE
print('Covariance: ', C)
ci_av=C_alpha*np.sqrt(var_av/N2) # semilength of confidence interval

print('Estimate MC '+str(mean_cmc)+' +- '+str(round(ci_cmc,5)))
print('Estimate AV '+str(mean_av)+' +- '+str(round(ci_av,5)))
print('Reduction '+str(round(ci_cmc/ci_av,5)))

# Plots of the last plot in memory and of its antithetic path
ax = plt.gca()
circle1 = plt.Circle((0, 0), r, color='darkturquoise', fill = True)  
circle2 = plt.Circle((0, 0), 2, color='darkturquoise', fill = False) 
circle3 = plt.Circle((0, 0), 3, color='darkturquoise', fill = False) 
circle4 = plt.Circle((0, 0), 4, color='darkturquoise', fill = False) 
circle5 = plt.Circle((0, 0), 5, color='darkturquoise', fill = False) 
ax.plot(X,Y,'forestgreen', linewidth = 0.5)
ax.plot(Xav,Yav,'red', linewidth = 0.5)
ax.plot(X[0],Y[0],'bo')
ax.plot(X[-1],Y[-1], 'go')
ax.plot(Xav[-1],Yav[-1],'ro')
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)
ax.add_patch(circle5)
plt.xlim(-parameters.xl,parameters.xl)
plt.ylim(-parameters.yl,parameters.yl)
ax.set_aspect('equal', adjustable='box')
