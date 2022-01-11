# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 21:17:21 2022

@author: giuli
"""

#Package Loading
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import parameters
import time

# Task
task = 'order'
# task = 'twostagesMC'

# Options
r = 1.0 # radius of the well
T = 1   # time limit
sigma = 2 # diffusion factor
Q = 1
alpha = 0.05 # confidence level 
C_alpha=st.norm.ppf(1-alpha/2) # quantile of a standard normal

# Setting seed for reproducibility
np.random.seed(21)

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

# Producing N trajectories    
def path(N):
    dt_c = 1e-2
    dt_f = dt_c/2
    count_f = 0
    count_c = 0
    Z_mc_f = np.zeros([N,])
    Z_mc_c = np.zeros([N,])
    for i in range(N): 
        # Starting value
        X0_f = parameters.X0
        Y0_f = parameters.Y0
        X0_c = parameters.X0
        Y0_c = parameters.Y0
        # Preparation of vectors where we store coordinates over all times until stop
        X_f = [X0_f]
        Y_f = [Y0_f]
        X_c = [X0_c]
        Y_c = [Y0_c]
        stop_f = False
        stop_c = False
        it = 0
        while(stop_f == False):
            # Generating two independent normals 
            Z1_f = np.float(np.random.normal(size = 1))
            Z2_f = np.float(np.random.normal(size = 1))
            Z1_c = Z1_f
            Z2_c = Z2_f
            
            # Computation of the following position
            new_X_f = X0_f + u1(X0_f,Y0_f)*dt_f + sigma * np.sqrt(dt_f)*Z1_f
            new_Y_f = Y0_f + u2(X0_f,Y0_f)*dt_f + sigma * np.sqrt(dt_f)*Z2_f
            
            # Storage of the next position
            X_f.append(new_X_f)
            Y_f.append(new_Y_f)
            
            if np.mod(it,2) == 0 :
                new_X_c = X0_c + u1(X0_c,Y0_c)*dt_c + sigma * np.sqrt(dt_c)*Z1_c
                new_Y_c = Y0_c + u2(X0_c,Y0_c)*dt_c + sigma * np.sqrt(dt_c)*Z2_c
                X_c.append(new_X_c)
                Y_c.append(new_Y_c)
                X0_c = new_X_c
                Y0_c = new_Y_c
                
                if (killing_boundary(new_X_c,new_Y_c)) : 
                    if stop_c == False:
                        count_c = count_c + 1
                        Z_mc_c[i] = 1
                        stop_c = True
                    
            # Checking wheter T is reached
            if (it > int(T/dt)) : stop_f = True
            # Checking whether the well is hit
            if (killing_boundary(new_X_f,new_Y_f)) : 
                stop_f = True
                count_f = count_f + 1
                Z_mc_f[i] = 1
  
            it = it + 1
            X0_f = new_X_f
            Y0_f = new_Y_f
    return count_f/N, X_f, Y_f, Z_mc_f, count_c/N, X_c, Y_c, Z_mc_c
    
start = time.time() # to keep track of time

if task == 'order':
    
    tol = parameters.tol # tolerance requested
    N = 10000 # N for pilot run
    dt = 1e-2/2 # temporal step for discretization
    mu_hat_fine, X_f, Y_f, Z_mc_hat_fine, mu_hat_coarse, X_c, Y_c, Z_mc_hat_coarse = path(N) # running N paths
    sigma_hat_coarse = np.var(Z_mc_hat_coarse) # computing empirical variance
    sigma_hat_fine = np.var(Z_mc_hat_fine)
    # print('Variance with old N: ', sigma_hat)
    # print('Extimated probability with old N: ', mu_hat)
    # print('New N: ', new_N)

    
    # print('Variance with new N: ', sig)
    # print('Extimated probability: ', mu)
    # print('Confidence Interval: [', mu, '+-', C_alpha * np.sqrt(sig)/np.sqrt(new_N),']')
    
    # if(sig < sigma_hat):
    #     print('Done')
    # else:
    #     print('Update N again')
    
    # end = time.time()
    
    # print("Simulation time: ", end - start, "s; in minutes: ",(end-start)/60) 
    
# Plots one path (the last in memory)
ax = plt.gca()
circle1 = plt.Circle((0, 0), r, color='darkturquoise', fill = True)  
ax.plot(X_f,Y_f,'forestgreen', linewidth = 0.5)
ax.plot(X_f[0],Y_f[0],'bo')
ax.plot(X_f[-1],Y_f[-1], 'ro')
ax.plot(X_c,Y_c,'cyan', linewidth = 0.5)
plt.xlim(-parameters.xl,parameters.xl)
plt.ylim(-parameters.yl,parameters.yl)
ax.set_aspect('equal', adjustable='box')
