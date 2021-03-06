# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:10:12 2021

@author: giuli
"""

"""
This file contains the code for the stochastic part of the first question of the assignment.
We propose a two-stage Monte Carlo for the estimation of the desired quantity.
Before launching the script, choose the task to perform: two stages MC or estimation of the order in time
"""

#Package Loading
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import parameters
import time

# Task
task = 'twostagesMC'
# task = 'order'

# Options
r = 1.0 # radius of the well
T = 1   # time limit
sigma = 2 # diffusion factor
Q = 1
alpha = 0.05 # confidence level 
C_alpha=st.norm.ppf(1-alpha/2) # corresponding quantile of the standard normal

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

# Function to produce N independent trajectories    
def path(N):
    count = 0 # to store how many trajectories hit the well
    Z_mc = np.zeros([N,]) # to store whether each trajectory leads to a hit or not
    for i in range(N): 
        # Starting values
        X0 = parameters.X0
        Y0 = parameters.Y0
        # Preparation of vectors where we store coordinates over all times until stop
        X = [X0]
        Y = [Y0]
        stop = False
        it = 0
        while(stop == False):
            # Generating two independent standard normals 
            Z1 = np.float(np.random.normal(size = 1))
            Z2 = np.float(np.random.normal(size = 1))
            # Computation of the following position
            new_X = X0 + u1(X0,Y0)*dt + sigma * np.sqrt(dt)*Z1
            new_Y = Y0 + u2(X0,Y0)*dt + sigma * np.sqrt(dt)*Z2
            # Storage of the next position
            X.append(new_X)
            Y.append(new_Y)
            # Checking wheter time limit T is reached
            if (it > int(T/dt)) : stop = True
            # Checking whether the well is hit
            if (killing_boundary(new_X,new_Y)) : 
                stop = True # stop the current trajectory
                count = count + 1 # update counter of success
                Z_mc[i] = 1 # register a success for the current trajectory
  
            it = it + 1
            # Update coordinates
            X0 = new_X
            Y0 = new_Y
            
    return count/N, X, Y, Z_mc
    
start = time.time() # to keep track of time

# TWO STAGES MONTE CARLO
if task == 'twostagesMC':
    
    tol = parameters.tol # tolerance requested
    N = 1000 # N for pilot run
    dt = 1e-2 # temporal step for discretization
    mu_hat, X, Y, Z_mc_hat = path(N) # running N paths
    sigma_hat = np.var(Z_mc_hat, ddof = 1) # computing empirical variance with the unbiased estimator
    new_N = int(np.ceil(C_alpha**2*sigma_hat/(tol/2)**2)) # computing the new N
    
    print('Variance with old N: ', sigma_hat)
    print('Extimated probability with old N: ', mu_hat)
    print('New N: ', new_N)
    
    mu, X, Y, Z_mc = path(new_N) # running new_N paths
    sigma_new = np.var(Z_mc, ddof = 1) # computing new variance with the unbiased estimator
    
    print('Variance with new N: ', sigma_new)
    print('Extimated probability: ', mu)
    print('Confidence Interval: [', mu, '+-', C_alpha * np.sqrt(sigma_new)/np.sqrt(new_N),']')
    
    if(sigma_new < sigma_hat):
        print('Done')
    else:
        print('Update N again')
        new_N = int(np.ceil(C_alpha**2*sigma_new/(tol/2)**2)) # computing the new N
        print('New N: ', new_N)
        
    end = time.time()
    
    print("Simulation time: ", end - start, "s; in minutes: ",(end-start)/60) 
    
    # Plots one path (the last in memory)
    ax = plt.gca()
    circle1 = plt.Circle((0, 0), r, color='darkturquoise', fill = True)  
    circle2 = plt.Circle((0, 0), 2, color='darkturquoise', fill = False) 
    circle3 = plt.Circle((0, 0), 3, color='darkturquoise', fill = False) 
    circle4 = plt.Circle((0, 0), 4, color='darkturquoise', fill = False) 
    circle5 = plt.Circle((0, 0), 5, color='darkturquoise', fill = False) 
    ax.plot(X,Y,'forestgreen', linewidth = 0.5)
    ax.plot(X[0],Y[0],'bo')
    ax.plot(X[-1],Y[-1], 'ro')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    ax.add_patch(circle5)
    plt.xlim(-parameters.xl,parameters.xl)
    plt.ylim(-parameters.yl,parameters.yl)
    ax.set_aspect('equal', adjustable='box')
    
# ORDER STUDY    
# Producing N trajectories with three different refinements on the same Brownian Paths   

def ref_path(N):
    # Time discretizations
    dt_c = 1e-2
    dt_f = dt_c/2
    dt_ff = dt_f/2
    dt_fff = dt_ff/2
    # Counters of hits
    count_c = 0
    count_f = 0
    count_ff = 0
    count_fff = 0
    # Storage of successes
    Z_mc_c = np.zeros([N,])
    Z_mc_f = np.zeros([N,])
    Z_mc_ff = np.zeros([N,])
    Z_mc_fff = np.zeros([N,])
    
    for i in range(N): 
        if np.mod(i,1000) == 0:
            print('Iter n. ', i)
        # Starting values
        X0_fff = parameters.X0
        Y0_fff = parameters.Y0
        X0_ff = parameters.X0
        Y0_ff = parameters.Y0
        X0_f = parameters.X0
        Y0_f = parameters.Y0
        X0_c = parameters.X0
        Y0_c = parameters.Y0
        # Preparation of vectors where we store coordinates over all times until stop (for the plot)
        X_fff = [X0_fff]
        Y_fff = [Y0_fff]
        X_ff = [X0_ff]
        Y_ff = [Y0_ff]
        X_f = [X0_f]
        Y_f = [Y0_f]
        X_c = [X0_c]
        Y_c = [Y0_c]
        # Boolean variables to stop the procedure
        stop_fff = False
        stop_ff = False
        stop_f = False
        stop_c = False
        # Initialization of counter for iterations
        it = 0
        # Cleaning brownian paths of medium-fine and coarse path
        W1_c = 0
        W2_c = 0
        W1_f = 0
        W2_f = 0
        W1_ff = 0
        W2_ff = 0
        while(stop_fff == False or stop_ff == False or stop_f == False or stop_c == False):
            
            it = it + 1 # incrementing the iteration counter 
            
            # Generating the Brownian path for the finest path
            W1_fff = np.sqrt(dt_fff)*np.float(np.random.normal(size = 1))
            W2_fff = np.sqrt(dt_fff)*np.float(np.random.normal(size = 1))
            # Accumulation the brownian path for the other two resolutions
            W1_ff = W1_ff + W1_fff
            W2_ff = W2_ff + W2_fff
            W1_f = W1_f + W1_fff
            W2_f = W2_f + W2_fff
            W1_c = W1_c + W1_fff
            W2_c = W2_c + W2_fff
            
            # FINEST PATH
            # Computation of the following position 
            new_X_fff = X0_fff + u1(X0_fff,Y0_fff)*dt_fff + sigma * W1_fff
            new_Y_fff = Y0_fff + u2(X0_fff,Y0_fff)*dt_fff + sigma * W2_fff
            
            # Storage of the position
            X_fff.append(new_X_fff)
            Y_fff.append(new_Y_fff)
            
            # Checking whether the well is hit
            if (killing_boundary(new_X_fff,new_Y_fff)) : 
                if stop_fff == False:
                    stop_fff = True
                    count_fff = count_fff + 1
                    Z_mc_fff[i] = 1
            # Updating the position        
            X0_fff = new_X_fff
            Y0_fff = new_Y_fff
            
            # SECOND FINEST PATH: update it every two iterations
            if np.mod(it,2) == 0 :
                X_ff,Y_ff,W1_ff,W2_ff,stop_ff,count_ff,Z_mc_ff = coarse_update(X_ff,Y_ff,dt_ff,W1_ff,W2_ff,stop_ff,count_ff,Z_mc_ff,i)
            
            # MEDIUM-FINE PATH: update it every four iterations
            if np.mod(it,4) == 0 :
               X_f,Y_f,W1_f,W2_f,stop_f,count_f,Z_mc_f = coarse_update(X_f,Y_f,dt_f,W1_f,W2_f,stop_f,count_f,Z_mc_f,i)
                    
            # COARSE PATH: update it every eight iterations
            if np.mod(it,8) == 0 :
                X_c,Y_c,W1_c,W2_c,stop_c,count_c,Z_mc_c = coarse_update(X_c,Y_c,dt_c,W1_c,W2_c,stop_c,count_c,Z_mc_c,i)
                        
            # Checking wheter time limit T is reached
            if (it > int(T/dt_fff)) : 
                stop_fff = True
                stop_ff= True
                stop_f = True 
                stop_c = True
                
        # # To plot the paths 
        # if Z_mc_fff[i] == 1:
        #     ax = plt.figure(i)
        #     ax = plt.gca()
        #     ax.plot(X_fff,Y_fff,'cyan', linewidth = 0.5, label = 'dt = 1.25e-3')
        #     ax.plot(X_ff,Y_ff,'magenta', linewidth = 0.5, label = 'dt = 2.5e-3')
        #     ax.plot(X_f[0],Y_f[0],'ro')
        #     ax.plot(X_f,Y_f,'lime', linewidth = 0.5, label = 'dt = 5e-3')
        #     ax.plot(X_f[-1],Y_f[-1], 'ro')
        #     ax.plot(X_c,Y_c,'blue', linewidth = 0.5, label = 'dt = 1e-2')
        #     ax.plot(X_c[-1],Y_c[-1], 'go')
        #     circle1 = plt.Circle((0, 0), r, color='darkturquoise', fill = True)  
        #     circle2 = plt.Circle((0, 0), 2, color='lightgray', fill = False) 
        #     circle3 = plt.Circle((0, 0), 3, color='lightgray', fill = False) 
        #     circle4 = plt.Circle((0, 0), 4, color='lightgray', fill = False) 
        #     circle5 = plt.Circle((0, 0), 5, color='lightgray', fill = False) 
        #     ax.add_patch(circle1)
        #     ax.add_patch(circle2)
        #     ax.add_patch(circle3)
        #     ax.add_patch(circle4)
        #     ax.add_patch(circle5)
        #     ax.legend()
        #     plt.xlim(-1,4)
        #     plt.ylim(0,3)
        #     ax.set_aspect('equal', adjustable='box')
            
    return count_fff/N, Z_mc_fff, count_ff/N, Z_mc_ff, count_f/N, Z_mc_f, count_c/N, Z_mc_c

# Function updating coarser paths
def coarse_update(X_c,Y_c,dt_c,W1_c,W2_c,stop_c,count_c,Z_mc_c,i):
    
    # Computation of the following position 
    new_X_c = X_c[-1] + u1(X_c[-1],Y_c[-1])*dt_c + sigma * W1_c
    new_Y_c = Y_c[-1] + u2(X_c[-1],Y_c[-1])*dt_c + sigma * W2_c
                
    # Storage of the position
    X_c.append(new_X_c)
    Y_c.append(new_Y_c)
                
    W1_c = 0 # cleaning W_f for the next iteration
    W2_c = 0 # cleaning W_f for the next iteration
    
    # Checking whether the well is hit    
    if (killing_boundary(new_X_c,new_Y_c)) : 
        if stop_c == False:
            stop_c = True
            count_c = count_c + 1
            Z_mc_c[i] = 1
    return X_c,Y_c,W1_c,W2_c,stop_c,count_c,Z_mc_c

if task == 'order':
    
    N = 10000 # Numer of replicas
    mu_hat_fff, Z_mc_hat_fff, mu_hat_ff, Z_mc_hat_ff, mu_hat_f, Z_mc_hat_f, mu_hat_c, Z_mc_hat_c = ref_path(N) # running N paths with the different resolutions
    
    end = time.time()
    
    print("Simulation time: ", end - start, "s; in minutes: ",(end-start)/60)
    
    # Variances
    sigma_hat_fff = np.var(Z_mc_hat_fff, ddof = 1)
    sigma_hat_ff = np.var(Z_mc_hat_ff, ddof = 1)
    sigma_hat_c = np.var(Z_mc_hat_c, ddof = 1) 
    sigma_hat_f = np.var(Z_mc_hat_f, ddof = 1)    
    
    # Extimated values and variances
    print('dt = 1.25e-3: mu_hat = ', mu_hat_fff, 'sigma_hat = ', sigma_hat_fff)
    print('dt = 2.5e-3: mu_hat = ', mu_hat_ff, 'sigma_hat = ', sigma_hat_ff)
    print('dt = 5e-3: mu_hat = ', mu_hat_f, 'sigma_hat = ', sigma_hat_f)
    print('dt = 1e-2: mu_hat = ', mu_hat_c, 'sigma_hat = ', sigma_hat_c)
    