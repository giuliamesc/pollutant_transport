# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:06:44 2022

@author: giuli
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Entrace in the circular crown with internal radius r and external radius R
def entrance(x,y,R):
    # This function receives the coordinate of a point and returns true if that coordinate is inside the circle of radius R
    radius = np.sqrt(np.power(x,2)+np.power(y,2))
    if(radius <= R) : return True
    else: return False
    
# Parameters
T = 1
sigma = 2
Q = 1
# Initial Position
X0 = 7.0
Y0 = 7.0
# Temporal Step
dt = 1e-3
# Maximum number of X,Y for a trajectory (stopping it at time T if it does not hit the well)
K = int(T/dt)

# Set seed for reproducibility
np.random.seed(21)

# Number of circles
circles = 7
# circles = 4

# Velocity functions
def u1(x,y): 
    return 1 + Q*x/(2*np.pi*(np.power(x,2)+np.power(y,2)))
def u2(x,y):
    return Q*y/(2*np.pi*(np.power(x,2)+np.power(y,2)))

# Generates a trajectory starting from X0,Y0, which ends after K steps or when the circle of radius R is hit
def rw(K,X0,Y0,R):
    finished=0 # flag for the stop of the chain
    x_new = X0 
    y_new = Y0
    xs = [X0] # storing the path - only for debugging, not necessary
    ys = [Y0] # storing the path - only for debugging, not necessary
    for i in range(K):   
        Z1 = np.float(np.random.normal(size = 1))
        Z2 = np.float(np.random.normal(size = 1))       
        # Velocity update
        u1_old = u1(x_new,y_new)
        u2_old = u2(x_new,y_new)
        # Move the trajectory
        x_new = x_new + u1_old*dt + sigma * np.sqrt(dt)*Z1
        y_new = y_new + u2_old*dt + sigma * np.sqrt(dt)*Z2
        xs.append(x_new)
        ys.append(y_new)
        if (entrance(x_new,y_new,R)) : 
            finished=1
            break; # exit from the for loop
           
    return finished,x_new,y_new,len(xs)

if circles == 7 :
    R = [7,6,5,4,3,2,1] # vector of radius
    #iters = [100,100,100,100,100,100,100] # vector of iterations to perform for each level (TRIAL)
    #P_trials = [0.08,0.04875,0.01564103,0.00704918]
    iters = [100,4,5,10,10,14,20] # vector of iterations to perform for each level

if circles == 4 :
    R = [7,5,3,1] # vector of radius
    #iters = [100,100,100,100] # vector of iterations to perform for each level (TRIAL)
    P_trials = [0.08,0.04875,0.01564103,0.00704918]
    iters = [100,25,64,142]

np.random.seed(21) # set seed for reproducibility

def splitting():
    X_start = [X0]
    Y_start = [Y0]
    times = [0] # to store after how many steps we hit the inner ball
    P = np.zeros(len(R)) # vector to store probabilities
    counts = np.zeros(len(R))
    for i in range(len(R)): # for each level
        #print("Level: ", i)
        its = int(iters[i]) # get how many iterations to perform for each valid starting point
        #print("Iterations to perform: ", its)
        den = its * len(X_start) # computing the denominator needed to divide P at the end: number of trials per starting point * number of starting points
        X_new = [] # I will store here the valid ending points found
        Y_new = [] 
        times_new = []
        #print("Number of valid starting points: ", len(X_start))
        for j in range(len(X_start)): # for each starting point
            t_old = times[j] # time employed to reach the current starting point
            # print('Times old: ', times[j])
            for k in range(its): 
                finished,x_new,y_new,temp_t = rw(K,X_start[j],Y_start[j],R[i]) # generate its time a trajectory starting from my starting point
                t_new = temp_t + t_old                
                if finished == 1 and t_new <= K :
                    X_new.append(x_new) # store the new initial position
                    Y_new.append(y_new)
                    times_new.append(t_new)
                    P[i] = P[i] + 1 # update counter
        #print('Number of hits: ', int(P[i]))
        counts[i] = P[i]            
        P[i] = P[i]/den # count the fraction of successes
        X_start = np.copy(X_new) # valid ending points become the new starting points
        Y_start = np.copy(Y_new)
        times = np.copy(times_new)
        #print('Finished Stage: ',i)
        return counts[-1], np.prod(P)

# Splitting
_,out = splitting()
print(out) # output    
    
# Variance Extimation
N = 10    
Rm = np.zeros(10,) # needed for variance calculation

for n in range(N):
    print('Iter n.:', n)
    Rm[n],_ = splitting()
    print('Hits:', Rm[n])
    
my_den = iters[0]**2
for i in np.arange(1,len(R)):
    my_den = my_den * (iters[i]**2)
variance = np.var(Rm)/my_den
print('Variance: ', variance)
# print('7.64490306122449e-11')
