'''
Created on Feb 22, 2020

@author: ryan
'''
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 7)
fig, ax = plt.subplots()

ax.plot(3.5, 1.8, "or", color="darkorange",
        markersize=15)
ax.plot(1.1, 3.9, "oy",
        markersize=15)

point_on_line = (4, 4.5)
ax.plot(1.1, 3.9, "oy", markersize=15)
#calculate gradient:
m = point_on_line[1] / point_on_line[0]
ax.plot(x, m * x, "g-", linewidth=3)
plt.show()