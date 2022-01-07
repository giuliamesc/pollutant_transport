# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 17:35:26 2021

@author: giuli
"""

# Variance Reduction with Antithetic Variables

import numpy as np
import scipy.stats as st
import parameters
import matplotlib.pyplot as plt


# Options and parameters
N=31398 # Number of samples
N2=int(N/2) # Number of samples for AV
dt = parameters.dt
T = 1
Q = 1
sigma = 2
r = 1.0
K = int(T/dt) # iteration corresponding to the desired stopping time
alpha = 0.05
C_alpha=st.norm.ppf(1-alpha/2)

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

# Defining the random walk for the CMC and the AV 
#----------------------------------------------------
def rw(K):
    
    finished_mc=0
    finished_av=0
    x_mc = parameters.X0
    y_mc = parameters.Y0
    x_av = parameters.X0
    y_av = parameters.Y0
    X = [x_mc]
    Y = [y_mc]
    Xav = [x_av]
    Yav = [y_av]
    
    for i in range(K):   
        Z1 = np.float(np.random.normal(size = 1))
        Z2 = np.float(np.random.normal(size = 1)) 
        stop_mc = False
        stop_av = False
        # Moves crude MC
        if stop_mc == False:
            u1_old = u1(x_mc,y_mc)
            u2_old = u2(x_mc,y_mc)
            x_mc = x_mc + u1_old*dt + sigma * np.sqrt(dt)*Z1
            y_mc = y_mc + u2_old*dt + sigma * np.sqrt(dt)*Z2
            X.append(x_mc)
            Y.append(y_mc)
            if (killing_boundary(x_mc,y_mc)) : 
                finished_mc=1
                stop_mc=True
        # Moving AV with the same random realization 
        if stop_av == False:        
            u1_old = u1(x_av,y_av)
            u2_old = u2(x_av,y_av)
            x_av = x_av + u1_old*dt - sigma * np.sqrt(dt)*Z1
            y_av = y_av + u2_old*dt - sigma * np.sqrt(dt)*Z2
            Xav.append(x_av)
            Yav.append(y_av)
            if (killing_boundary(x_av,y_av)) : 
                finished_av=1
                stop_av = True
           
    return finished_mc,finished_av,X,Y,Xav,Yav

for i in range(int(N)):    
    Z_mc[i],_,_,_,_,_=rw(K)


mean_cmc=np.mean(Z_mc)
var_cmc=np.var(Z_mc)
print('Variance CMC: ', var_cmc)
ci_cmc=C_alpha*np.sqrt(var_cmc)/np.sqrt(N)

# Compute the AV estimator 
for i in range(int(N2)):    
    Z_mc2[i],Z_av[i],X,Y,Xav,Yav=rw(K)

Y_av=0.5*(Z_mc2+Z_av)
print('Variance AV: ', np.var(Y_av))
mean_av=np.mean(Y_av)
C=np.cov(Z_mc2,Z_av)
ci_av=C_alpha*np.sqrt((np.sum(C))/(2*N))

#Print results
print('--------------------------------------------------')
print('mean MC '+str(mean_cmc)+' +- '+str(round(ci_cmc,5)))
print('mean AV '+str(mean_av)+' +- '+str(round(ci_av,5)))
print('Reduction '+str(round(ci_cmc/ci_av,5)))

# Plots
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
