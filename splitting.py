# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 08:06:44 2022

@author: giuli
"""

"""
This script contains the code to solve the third question of the assignment.
We implemented the splitting method in two variants (Fixed Effort and Fixed Splitting), and computed the variance for Fixed Splitting.
Before launching the script, choose the task to perform and the number of circular crowns.
"""

# Package Loading
import numpy as np

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

# Maximum number of X,Y for a trajectory (to stop it at time T if it does not hit the well)
K = int(T/dt)

# Set seed for reproducibility
np.random.seed(21)

# CHOOSE HERE THE NUMBER OF CIRCULAR CROWNS
# Number of circles
circles = 7
# circles = 4

# CHOOSE HERE THE TASK TO PERFORM
# Task
task = 'FSestimation' # fixed splitting, estimation 
# task = 'FSvariance' # fixed splitting, variance 
# task = 'FEestimation' # fixed effort, estimation

# Velocity functions
def u1(x,y): 
    return 1 + Q*x/(2*np.pi*(np.power(x,2)+np.power(y,2)))
def u2(x,y):
    return Q*y/(2*np.pi*(np.power(x,2)+np.power(y,2)))

# This function generates a trajectory starting from (X0,Y0), which ends after K steps or when the circle of radius R is hit
def subpath(K,X0,Y0,R):
    finished = 0 # flag for the stop of the path
    x_new = X0 # starting position
    y_new = Y0 # starting position
    xs = [X0] # storing the path - only for debugging, not necessary
    ys = [Y0] # storing the path - only for debugging, not necessary
    for i in range(K):   
        # Generate two independent standard normals
        Z1 = np.float(np.random.normal(size = 1))
        Z2 = np.float(np.random.normal(size = 1))       
        # Velocity update
        u1_old = u1(x_new,y_new)
        u2_old = u2(x_new,y_new)
        # Move the trajectory
        x_new = x_new + u1_old*dt + sigma * np.sqrt(dt)*Z1
        y_new = y_new + u2_old*dt + sigma * np.sqrt(dt)*Z2
        # Store the new position
        xs.append(x_new)
        ys.append(y_new)
        # Check if we hit next stage
        if (entrance(x_new,y_new,R)) : 
            finished = 1
            break; # exit from the for loop
           
    return finished,x_new,y_new,len(xs)

if circles == 7 :
    R = [7,6,5,4,3,2,1] # vector of radius
    #iters = [100,100,100,100,100,100,100] # vector of iterations to perform for each level (TRIAL)
    iters = [100,4,5,10,10,14,20] # vector of iterations to perform for each level

if circles == 4 :
    R = [7,5,3,1] # vector of radius
    #iters = [100,100,100,100] # vector of iterations to perform for each level (TRIAL)
    iters = [100,25,64,142] # vector of iterations to perform for each level

# Splitting
def splitting(task):
    X_start = [X0] # vector of initial positions
    Y_start = [Y0] # vector of initial positions
    times = [0] # to store after how many steps we hit the inner ball
    P = np.zeros(len(R)) # vector to store the hitting probabilities
    counts = np.zeros(len(R)) # vector to store the counter of hits
    parents = [] # to store "parents" - for variance estimate, we need to keep track of the initial path from which each successful path originated
    for i in range(len(R)): # for each level
        print("Level: ", i)
        if task == 'FSestimation' or task == 'FSvariance' :
            its = int(iters[i]) # get how many iterations to perform for each valid starting point
        if task == 'FEestimation':
            its =  int(N_fe[i] / len(X_start)) # FE choice of iterations
        print("Iterations to perform: ", its)
        den = its * len(X_start) # computing the denominator needed to divide P at the end: number of trials per starting point * number of starting points
        X_new = [] # we will store here the valid ending points found
        Y_new = [] 
        times_new = [] # ...their total time for reaching the next stage
        parents_new = [] #...their parents
        print("Number of valid starting points: ", len(X_start))
        for j in range(len(X_start)): # for each starting point
            t_old = times[j] # time employed to reach the current starting point
            if i > 0: # only after the first iteration...
                parents_old = parents[j] # inheritage of the parent from the starting point
            for k in range(its): 
                if i == 0 : # in the first iteration, we identify the iters[0] "parents"
                    parents_old = k
                finished,x_new,y_new,temp_t = subpath(K,X_start[j],Y_start[j],R[i]) # generate each time a trajectory starting from our starting point
                t_new = temp_t + t_old # updating the arrival time         
                if finished == 1 and t_new <= K :
                    X_new.append(x_new) # store the new initial position
                    Y_new.append(y_new)
                    times_new.append(t_new) # ... its time of arrival
                    parents_new.append(parents_old) # ... its "parent"
                    P[i] = P[i] + 1 # update counter
        print('Number of hits: ', int(P[i]))
        counts[i] = P[i]            
        P[i] = P[i]/den # count the fraction of successes
        X_start = np.copy(X_new) # valid ending points become the new starting points
        Y_start = np.copy(Y_new)
        times = np.copy(times_new)
        parents = np.copy(parents_new)
        print('Finished Stage: ',i)
    return counts[-1], np.prod(P), parents

# Fixed Splitting
if task == 'FSestimation' :
    _,out,_ = splitting(task)
    print('Estimated probability: ', out)

# Variance Extimation
if task == 'FSvariance' :
    
    # CHOOSE HERE THE METHOD TO ESTIMATE VARIANCE -- see report
    method = 'Y1'
    # method = 'Rm'
    
    if method == 'Y1':
        # Denominator 
        my_den = iters[0]
        for i in np.arange(1,len(R)):
            my_den = my_den * (iters[i]**2)
        # Getting the parents and the number of success for successful hits    
        _,_,pars = splitting(task)
        counts = [(i,list(pars).count(i)) for i in set(pars)]
        # Transforming the output into an array containing the number of successes for each parent
        hits = np.zeros(iters[0])
        for i in range(len(counts)):
            hits[counts[i][0]] = counts[i][1]
        # Variance computation
        variance = np.var(hits,ddof=1)/my_den
        
    if method == 'Rm':
        N = 100
        Rm = np.zeros(N,) # to store number of successful hits
        # Running N times the algorithm
        for n in range(N):
            print('Iter n.:', n)
            Rm[n],_,_ = splitting(task)
            print('Hits:', Rm[n])
        # Denominator
        my_den = iters[0]**2
        for i in np.arange(1,len(R)):
            my_den = my_den * (iters[i]**2)
        # Variance computation    
        variance = np.var(Rm, ddof = 1)/my_den

    print('Variance: ', variance)
    print('Standard Deviation: ', np.sqrt(variance))

# Fixed Effort
if task == 'FEestimation' :
    N_fe = 1000*np.ones(len(R)) # fixing effort to 1000
    _,out,_ = splitting(task)
    print('Estimated probability: ', out)
    
