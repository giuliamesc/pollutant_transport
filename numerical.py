# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 08:47:37 2021

@author: giuli
"""
# Numerical solution of the parabolic PDE

import dolfin as df
import numpy as np
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
# import ffc
# print(ffc.supported_elements)
# Options
r = 1.0
R = 50.0
dt  = 1e-3
T = 1
sigma = 2
X0 = 3.0
Y0 = 4.0

tol_int = 0.15
tol_ext = 1e-2

# Mesh Generation
domain = Circle(df.Point(0, 0), R) - Circle(df.Point(0,0), r)
mesh = generate_mesh(domain, 200)
mesh_points=mesh.coordinates()
xm=mesh_points[:,0]
ym=mesh_points[:,1]
ax = plt.gca()
circle1 = plt.Circle((0, 0), R, color='red', fill = False)
ax.add_patch(circle1)
circle2 = plt.Circle((0, 0), r, color='red', fill = False)
ax.add_patch(circle2)
plt.plot(xm,ym,'bo')
plt.savefig('debug.png')
# Importing the Mesh in GMSH for visualization
xdmf = XDMFFile("mesh.xdmf")
xdmf.write(mesh)
xdmf.close()
import meshio
meshio_mesh = meshio.read("mesh.xdmf")
meshio.write("meshio_mesh.msh", meshio_mesh)

# Data Storage
def save_output(u, t, it):
    u.rename('u', 'u')
    xdmf_file = df.XDMFFile('data/UnsteadyCase/unsteady_%05d.xdmf' % it)
    xdmf_file.parameters['flush_output'] = True
    xdmf_file.parameters['functions_share_mesh'] = True
    xdmf_file.parameters['rewrite_function_mesh'] = False
    xdmf_file.write(u, t)
    
V = df.FunctionSpace(mesh,"Lagrange",1)

# BC
bcs = list()
  

def my_well(x,on_boundary):
        rad = np.sqrt(np.power(x[0],2)+np.power(x[1],2))
        return on_boundary and (np.abs(rad - r) < tol_int)
    
def my_out(x,on_boundary):
        rad = np.sqrt(np.power(x[0],2)+np.power(x[1],2))
        return on_boundary and (np.abs(rad - R) < tol_ext)

bcs.append(df.DirichletBC(V, df.Constant(1), my_well))
bcs.append(df.DirichletBC(V, df.Constant(0), my_out))

# Test and Trial Space
# Test and Trial Functions in the bilinear form
v = df.TestFunction(V)
phi = df.TrialFunction(V)

# # Point generation
# num_out_points = 1000
# rhos = np.random.uniform(r,R,num_out_points)
# thetas = np.random.uniform(0,2*np.pi,num_out_points)
# x = rhos*np.cos(thetas)
# y = rhos*np.sin(thetas)
# ax = plt.gca()
# ax.plot(x,y,'ro')
# ax.set_aspect('equal', adjustable='box')

# Velocity field expression and interpolation
u1 = df.Expression('1+x[0]/(2*3.14*(std::pow(x[0],2)+std::pow(x[1],2)))', degree = 2)
u2 = df.Expression('x[1]/(2*3.14*(std::pow(x[0],2)+std::pow(x[1],2)))', degree = 2)
interpolate(u1,V)
interpolate(u2,V)


print('Giulia')

# Time discretization
times = np.arange(T, -0.01, step = -dt)
times[-1] = 0
phi_old = interpolate(df.Constant(0),V) # initial condition

save_output(phi_old, T, 0)


for i in range(1, len(times)):
    t = times[i]
    if(np.mod(i,10)==0):
        print('*** Solving time t = %1.6f ***' % t)
    
    # Bilinear form
    a = (
        (phi*v) / df.Constant(dt) 
        - df.Constant(sigma**2/2) * (df.div(df.grad(phi))*v) 
        + (u1*phi.dx(0)+u2*phi.dx(1))*v
        )*df.dx
    rhs = ((phi_old*v)/df.Constant(dt))*df.dx
    solve(a == rhs, phi_old, bcs=bcs)
    #solve(a == rhs, phi_old, bcs = [])
    save_output(phi_old, t, i)

    
print(np.array(phi_old(X0,Y0)))
print('Done')