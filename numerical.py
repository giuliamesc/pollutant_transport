# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 08:47:37 2021

@author: giuli
"""
"""
This file contains the script for the numerical solution of the backwards parabolic PDE.
"""

# Package Loading
import dolfin as df
import numpy as np
import mshr
import parameters

# Options
r = 1.0 # inner radius
R = 10.0 # outer radius
dt  = 1e-2 # time discretization
T = 1 # time limit
sigma = 2 # diffusion factor
Q = 1

# Starting Position
X0 = parameters.X0
Y0 = parameters.Y0

tol_int = 1.0 # tolerance for inner boundary
tol_ext = 1e-1 # tolerance for outer boundary

# Domain Generation
domain = mshr.Circle(df.Point(0, 0), R) - mshr.Circle(df.Point(0,0), r)
# Mesh Generation
mesh = mshr.generate_mesh(domain,100)

# Importing the Mesh in GMSH for visualization
xdmf = df.XDMFFile("mesh.xdmf")
xdmf.write(mesh)
xdmf.close()
import meshio
meshio_mesh = meshio.read("mesh.xdmf")
meshio.write("meshio_mesh.msh", meshio_mesh)

# Data Storage for visualization in ParaView
def save_output(u, t, it):
    u.rename('u', 'u')
    xdmf_file = df.XDMFFile('data/100times/time_%05d.xdmf' % it)
    xdmf_file.parameters['flush_output'] = True
    xdmf_file.parameters['functions_share_mesh'] = True
    xdmf_file.parameters['rewrite_function_mesh'] = False
    xdmf_file.write(u, t)

# Finite Elements Space: P1    
V = df.FunctionSpace(mesh,"Lagrange",1)

# BCs
bcs = list()

# Boundary Definition      
def my_well(x,on_boundary):
        rad = np.sqrt(np.power(x[0],2)+np.power(x[1],2))
        return on_boundary and (np.abs(rad - r) < tol_int)

def my_out(x,on_boundary):
        rad = np.sqrt(np.power(x[0],2)+np.power(x[1],2))
        return on_boundary and (np.abs(rad - R) < tol_ext)
    
# Dirichlet Boundary Conditions
bcs.append(df.DirichletBC(V, df.Constant(1), my_well))
bcs.append(df.DirichletBC(V, df.Constant(0), my_out))

# Test and Trial Functions in the bilinear form
v = df.TestFunction(V)
phi = df.TrialFunction(V)

# Velocity field
u1 = df.Expression('1+Q*x[0]/(2*pi*(std::pow(x[0],2)+std::pow(x[1],2)))', Q=Q, pi = np.pi, degree = 2)
u2 = df.Expression('Q*x[1]/(2*pi*(std::pow(x[0],2)+std::pow(x[1],2)))', Q=Q, pi = np.pi, degree = 2)

# Time discretization
N = int(T/dt)
times = np.linspace(0, T, N)
times = times[::-1]

phi_old = df.interpolate(df.Constant(0),V) # final condition
save_output(phi_old, T, 0) # storing the final condition

for i in range(1, len(times)):
    t = times[i]
    if(np.mod(i,10)==0):
        print('*** Solving time t = %1.6f ***' % t)
    print(np.array(phi_old(X0,Y0)))
    # Bilinear form and right-hand side
    F = (- phi*v - df.Constant(dt) * df.Constant(sigma**2/2) * df.inner(df.grad(phi), df.grad(v)) + df.Constant(dt) * (df.inner(df.as_vector((u1,u2)), df.nabla_grad(phi)))*v + phi_old * v ) * df.dx
    a, rhs = df.lhs(F), df.rhs(F)
    # Solver
    df.solve(a == rhs, phi_old, bcs=bcs)
    # Storage of the solution
    save_output(phi_old, t, i)
    
print(np.array(phi_old(X0,Y0)))
print('Done')