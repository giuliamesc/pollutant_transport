# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 08:47:37 2021

@author: giuli
"""
# Numerical solution of the parabolic PDE

import dolfin as df
import numpy as np
from mshr import *

# Options
R = 5.0
T  = 1e-2


# Mesh
domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 64)
plot(mesh)
