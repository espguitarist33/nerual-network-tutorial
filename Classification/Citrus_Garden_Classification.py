'''
Created on Feb 22, 2020

@author: ryan
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
from random import shuffle
slope = 0.1

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

fruits = list(zip(oranges_x,
                  oranges_y,
                  repeat(0, len(oranges_x))))
fruits += list(zip(lemons_x,
                   lemons_y,
                   repeat(1, len(oranges_x))))

shuffle(fruits)

learning_rate = 0.2

line = None
counter = 0
for x, y, label in fruits:
    res = slope * x - y
    if label == 0 and res < 0:
        # Point is above the line but should be below, increment slope
        slope += learning_rate
        counter +=1
        ax.plot(X, slope * X,
                linewidth=2, label=str(counter))
        
    elif label == 1 and res > 1:
        # point is below but should be above
        slope -= learning_rate
        counter += 1
        ax.plot(X, slope * X,
                linewidth=2, label=str(counter))
        

    

ax.legend()
ax.grid()
plt.show()