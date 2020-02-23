'''
Created on Feb 22, 2020

@author: ryan
'''

import numpy as np
import matplotlib.pyplot as plt

def points_within_circle(radius,
                         center=(0,0),
                         number_of_points=100):
    center_x, center_y = center
    r = radius * np.sqrt(np.random.random((number_of_points,)))
    theta = np.random.random((number_of_points,)) * 2 * np.pi
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    return x,y

X = np.arange(0,8)
fig, ax = plt.subplots()
oranges_x, oranges_y = points_within_circle(1.6, (5,2), 100)
lemons_x, lemons_y = points_within_circle(1.9, (2,5), 100)

ax.scatter(oranges_x,
           oranges_y,
           c="orange",
           label="oranges")
ax.scatter(lemons_x,
           lemons_y,
           c="y",
           label="lemons")

ax.plot(X, 0.9 * X, "g-", linewidth=2)

ax.legend()
ax.grid()
plt.show()