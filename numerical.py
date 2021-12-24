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

# Options
r = 1.0
R = 5.0
dt  = 1e-2
T = 1
sigma = 2

# Mesh Generation
domain = Circle(df.Point(0, 0), R) - Circle(df.Point(0,0), r)
mesh = generate_mesh(domain, 64)

# Importing the Mesh in GMSH for visualization
xdmf = XDMFFile("mesh.xdmf")
xdmf.write(mesh)
xdmf.close()
import meshio
meshio_mesh = meshio.read("mesh.xdmf")
meshio.write("meshio_mesh.msh", meshio_mesh)

# Boundary Parts
boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
well = CompiledSubDomain('near((std::pow(x[0],2)+std::pow(x[1],2)), r)', r=np.sqrt(r))
out = CompiledSubDomain('near((std::pow(x[0],2)+std::pow(x[1],2)), R)', R=np.sqrt(R))
well.mark(boundary_parts, 0)
out.mark(boundary_parts, 1)

# Solver
V = df.FunctionSpace(mesh, 'P', 1)

# BC
bcs = list()
tol = 1e-14

def well_boundary(x, on_boundary):
        return on_boundary and (np.sqrt(x[0]**2+x[1]**2) < r - tol)
    
def domain_boundary(x, on_boundary):
        return on_boundary and (np.sqrt(x[0]**2+x[1]**2) > R - tol)
    
bcs.append(df.DirichletBC(V, df.Constant(1), well_boundary))
bcs.append(df.DirichletBC(V, df.Constant(0), domain_boundary))

# Test and Trial Space
v = df.TestFunctions(V)
phi = df.TrialFunctions(V)


# Point generation
num_out_points = 10000 
rhos = np.random.uniform(r,R,num_out_points)
thetas = np.random.uniform(0,2*np.pi,num_out_points)
x = rhos*np.cos(thetas)
y = rhos*np.sin(thetas)
plt.plot(x,y,'ro')

# Velocity field expression and interpolation
uu = df.Expression('1+x[0]/(2*pi*(std::pow(x[0],2)+std::pow(x[1],2)))','x[1]/(2*pi*(std::pow(x[0],2)+std::pow(x[1],2)))', pi = np.pi, degree = 2)
print('Giulia')

# Time discretization
phi = df.Function(V)
times = np.arange(T, 0.0, step = -dt)
phi_old = df.Constant(1) # initial condition

# For the line integral of the RHS
dS = Measure('dS', domain = mesh, subdomain_data = boundary_parts)

for i in range(1, len(times)):
    t = times[i]
    print('*** solving time t = %1.6f ***' % t)
    
    # Bilinear form
    a = (df.inner(phi - phi_old, v)/df.Constant(dt) - df.Constant(sigma**2/2) * df.inner(df.grad(phi), df.grad(v)) - df.inner(uu,df.grad(v))*phi)*df.dx
        
    # RHS
    rhs = (- df.inner(uu*df.Constant(1,1))*v)*dS(0)
    solve(a == rhs, phi, bcs)
    
    # Updating 
    phi_old.assign(phi)
    
print(phi)