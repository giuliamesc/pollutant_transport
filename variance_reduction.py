# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 17:35:26 2021

@author: giuli
"""

# Variance Reduction with Antithetic Variables

import numpy as np
import scipy.stats as st
import parameters

N=10000 #Number of samples
N2=int(N/2) #Number of samples for AV
Z_mc=np.zeros(N)
Z_mc2=np.zeros(N2)
Z_av=np.zeros(N2)
dt = 1e-2
T = 1
Q = 1
sigma = 2
maxit = 1000
K = int(T/dt) # iteration corresponding to the desired stopping time
alpha = 0.05
C_alpha=st.norm.ppf(1-alpha/2)

# Velocity functions
def u1(x,y): 
    return 1 + Q*x/(2*np.pi*(np.power(x,2)+np.power(y,2)))

def u2(x,y):
    return Q*y/(2*np.pi*(np.power(x,2)+np.power(y,2)))


def killing_boundary(x,y):
    # This function receives the coordinate of a point and returns true if that coordinate is inside the well
    if(np.sqrt(np.power(x,2)+np.power(y,2)) <= 1) : return True
    else: return False

# Defining the random walk for the CMC and the AV 
#----------------------------------------------------
def rw(maxit,K):
    
    finished_mc=0
    finished_av=0
    x_mc = parameters.X0
    y_mc = parameters.Y0
    x_av = parameters.X0
    y_av = parameters.Y0
    
    for i in range(maxit):   
        
        Z1 = np.float(np.random.normal(size = 1))
        Z2 = np.float(np.random.normal(size = 1))       
        # Moves crude MC
        u1_old = u1(x_mc,y_mc)
        u2_old = u2(x_mc,y_mc)
        x_mc = x_mc + u1_old*dt + sigma * np.sqrt(dt)*Z1
        y_mc = y_mc + u2_old*dt + sigma * np.sqrt(dt)*Z2
        if (killing_boundary(x_mc,y_mc)) : finished_mc=1
         # Moving AV with the same random realization         
        u1_old = u1(x_av,y_av)
        u2_old = u2(x_av,y_av)
        x_av = x_av + u1_old*dt - sigma * np.sqrt(dt)*Z1
        y_av = y_av + u2_old*dt - sigma * np.sqrt(dt)*Z2
        if (killing_boundary(x_av,y_av)) : finished_av=1

           
    return finished_mc,finished_av

for i in range(int(N)):    
    Z_mc[i],_=rw(maxit,K)


mean_cmc=np.mean(Z_mc)
var_cmc=np.var(Z_mc)
ci_cmc=C_alpha*np.sqrt(var_cmc)/np.sqrt(N)

# Compute the AV estimator 
for i in range(int(N2)):    
    Z_mc2[i],Z_av[i]=rw(maxit,K)

Y_av=0.5*(Z_mc2+Z_av)
mean_av=np.mean(Y_av)
C=np.cov(Z_mc2,Z_av)
ci_av=C_alpha*np.sqrt( (np.sum(C))/(2*N) )

#Print results
print('--------------------------------------------------')
print('mean MC '+str(mean_cmc)+' +- '+str(round(ci_cmc,5)))
print('mean AV '+str(mean_av)+' +- '+str(round(ci_av,5)))
print('Reduction '+str(round(ci_cmc/ci_av,5)))
