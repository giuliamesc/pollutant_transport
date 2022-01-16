# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 18:55:43 2022

@author: giuli
"""

import matplotlib.pyplot as plt

ax = plt.gca()
circle1 = plt.Circle((0, 0), 1, color='darkturquoise', fill = True)  
circle2 = plt.Circle((0, 0), 2, color='darkturquoise', fill = False) 
circle3 = plt.Circle((0, 0), 3, color='darkturquoise', fill = False) 
circle4 = plt.Circle((0, 0), 4, color='darkturquoise', fill = False) 
circle5 = plt.Circle((0, 0), 5, color='darkturquoise', fill = False) 
circle6 = plt.Circle((0, 0), 5, color='darkturquoise', fill = False) 
circle7 = plt.Circle((0, 0), 6, color='darkturquoise', fill = False) 
circle8 = plt.Circle((0, 0), 7, color='darkturquoise', fill = False) 
circle9 = plt.Circle((0, 0), 8, color='darkturquoise', fill = False) 
circle10 = plt.Circle((0, 0), 9, color='darkturquoise', fill = False) 
circle11 = plt.Circle((0, 0), 10, color='darkturquoise', fill = False) 

ax.plot(1.2,1.1,'o', label = '(1.2,1.1)')
ax.plot(2.5,2.5, 'o', label = '(2.5,2.5)')
ax.plot(-1.2,1.1, 'o', label = '(-1.2,1.1)')
ax.plot(3,4, 'o', label = '(3,4)')
ax.plot(7,7, 'o', label = '(7,7)')
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)
ax.add_patch(circle5)
ax.add_patch(circle6)
ax.add_patch(circle7)
ax.add_patch(circle8)
ax.add_patch(circle9)
ax.add_patch(circle10)
ax.add_patch(circle11)
plt.xlim(-10,10)
plt.ylim(-10,10)
ax.legend()
ax.set_aspect('equal', adjustable='box')