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
dt  = 1e-1
T = 1
sigma = 2
X0 = 2.5
Y0 = 2.5

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

# Data Storage
def save_output(u, t, it):
    u.rename('u', 'u')
    xdmf_file = df.XDMFFile('data/UnsteadyCase/unsteady_%05d.xdmf' % it)
    xdmf_file.parameters['flush_output'] = True
    xdmf_file.parameters['functions_share_mesh'] = True
    xdmf_file.parameters['rewrite_function_mesh'] = False
    xdmf_file.write(u, t)
    
# import pandas as pd

# data_list = []

# def create_csv_for_df(u, t):
#     u_points = np.array([u(i,j) for j in y for i in x])
#     x_tab = np.array([x for j in y for i in x])
#     y_tab = np.array([y for j in y for i in x])
#     t_tab = np.array([t for j in y for i in x])
#     data = pd.DataFrame({'t': t_tab,
#                          'x': x_tab,
#                          'y': y_tab,
#                          'ux': u_points[:,0]})
#     return data
    
# Boundary Parts
boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
well = CompiledSubDomain('near((std::pow(x[0],2)+std::pow(x[1],2)), r)', r=np.sqrt(r))
out = CompiledSubDomain('near((std::pow(x[0],2)+std::pow(x[1],2)), R)', R=np.sqrt(R))
well.mark(boundary_parts, 0)
out.mark(boundary_parts, 1)

# Solver
Q = df.FiniteElement("CG", mesh.ufl_cell(), 1)
V = df.FunctionSpace(mesh, Q)
VV = df.FunctionSpace(mesh,Q*Q)

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
# Test and Trial Functions in the bilinear form
v = df.TestFunction(V)
phi = df.TrialFunction(V)

# Point generation
num_out_points = 1000 
rhos = np.random.uniform(r,R,num_out_points)
thetas = np.random.uniform(0,2*np.pi,num_out_points)
x = rhos*np.cos(thetas)
y = rhos*np.sin(thetas)
ax = plt.gca()
ax.plot(x,y,'ro')
ax.set_aspect('equal', adjustable='box')

# Velocity field expression and interpolation
#uu = df.Expression('1+x[0]/(2*pi*(std::pow(x[0],2)+std::pow(x[1],2)))','x[1]/(2*pi*(std::pow(x[0],2)+std::pow(x[1],2)))', pi = np.pi, degree = 2)
#interpolate(uu,VV)
print('Giulia')

# Time discretization
times = np.arange(T, -0.01, step = -dt)
times[-1] = 0
phi_old = interpolate(df.Constant(1),V) # initial condition

# For the line integral of the RHS
dS = Measure('dS', domain = mesh, subdomain_data = boundary_parts)

save_output(phi_old, 0, 0)
# temp_dataframe = create_csv_for_df(phi_old, 0)
# data_list.append(temp_dataframe)

for i in range(1, len(times)):
    t = times[i]
    print('*** solving time t = %1.6f ***' % t)
    
    # Bilinear form
    a = (
        df.inner(phi, v)/df.Constant(dt) 
        - df.Constant(sigma**2/2) * df.inner(df.grad(phi), df.grad(v)) 
        #- df.inner(uu,df.grad(v))*phi
        )*df.dx
    #rhs = - df.inner(uu,df.Constant(1,1))*v*dS(0) + df.inner(phi_old,v)/df.Constant(dt)*df.dx
    rhs = df.inner(phi_old,v)/df.Constant(dt)*df.dx
    solve(a == rhs, phi_old, bcs=bcs)
    save_output(phi_old, t, i)
    # temp_dataframe = create_csv_for_df(phi_old, t)
    # data_list.append(temp_dataframe)
    
# dataframe_output_file = 'data/UnsteadyCase/unsteady_r'
# output_dataframe = pd.concat(data_list)
# output_dataframe.to_csv(dataframe_output_file + '.csv', index = False)    
print(np.array(phi_old(X0,Y0)))
print('Done')