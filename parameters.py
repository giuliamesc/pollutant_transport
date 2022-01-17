# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:57:16 2021

@author: giuli
"""

# Parameters file
testcase = 2

if testcase == 1:
    # Starting point
    X0 = 1.2
    Y0 = 1.1
    # Plot limits
    xl = 3
    yl = 3
    #Tolerance
    tol = 1e-2

if testcase == 2:
    # Starting point
    X0 = 2.5
    Y0 = 2.5
    # Plot limits
    xl = 5.5
    yl = 5.5
    #Tolerance
    tol = 5e-3

if testcase == 3:
    # Starting point
    X0 = 3
    Y0 = 4
    # Plot limits
    xl = 7
    yl = 7
    # Tolerance
    tol = 1e-4
    
if testcase == 4:
    # Starting point
    X0 = -1.2
    Y0 = 1.1
    # Plot limits
    xl = 3
    yl = 3
    # Tolerance
    tol = 1e-2
    
if testcase == 5:
    # Starting point
    X0 = 7
    Y0 = 7
    # Plot limits
    xl = 10
    yl = 10
    # Tolerance
    tol = 1e-1 # we need to specify a value to make the simulation script run; the output will always be 0.0