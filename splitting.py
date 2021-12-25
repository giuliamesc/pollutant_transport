# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 09:12:44 2021

@author: giuli
"""
import numpy as np
import matplotlib.pyplot as plt

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


R = np.arange(8.5,0.5,-0.5) # vector of external radius of my circular crowns
r = R-0.5 
r[-1] = 0 # vector of internal radius of my circular crowns
rho = np.zeros(len(R))
for i in range(len(R)-1):
    rho[i+1] = np.random.uniform(r[i],R[i],size=(1,))
theta = np.random.uniform(0,2*np.pi,size = len(R))
P = np.zeros(len(R)) # vector to store the probabilities P(tau_j <= T | tau_(j-1) <= T)
X = rho * np.cos(theta)
Y = rho * np.sin(theta)
X[0] = X0
Y[0] = Y0

def rw(K,X0,Y0,r,R):
    
    finished_mc=0
    finished_av=0
    x_mc = X0
    y_mc = Y0
    
    for i in range(K):   
        Z1 = np.float(np.random.normal(size = 1))
        Z2 = np.float(np.random.normal(size = 1))       
        # Moves crude MC
        u1_old = u1(x_mc,y_mc)
        u2_old = u2(x_mc,y_mc)
        x_mc = x_mc + u1_old*dt + sigma * np.sqrt(dt)*Z1
        y_mc = y_mc + u2_old*dt + sigma * np.sqrt(dt)*Z2
        if (entrance(x_mc,y_mc,r,R)) : 
            finished_mc=1
           
    return finished_mc

for j in range(len(R)):
    Z = 0
    for k in range(int(N)):    
        Z = Z + rw(K,X[j],Y[j],r[j],R[j])
    Z = Z / N
    P[j] = Z
    

ax = plt.gca()
circle1 = plt.Circle((0, 0), R[-1], color='darkturquoise', fill = True)
ax.add_patch(circle1)
for i in range(len(R)-1):
    circle = plt.Circle((0, 0), R[i], color='green', fill = False)
    ax.add_patch(circle)
plt.xlim(-9,9)
plt.ylim(-9,9)
ax.set_aspect('equal', adjustable='box')
ax.plot(X,Y,'b*')
ax.plot(X[0],Y[0],'bo')
ax.plot(X[-1],Y[-1], 'ro')

res = np.prod(P)
print(res)
