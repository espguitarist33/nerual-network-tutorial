'''
Created on Feb 22, 2020

@author: ryan
'''
import numpy as np
from collections import Counter
from builtins import staticmethod


def points_within_circle(radius,
                         center=(0,0),
                         number_of_points=100):
    center_x, center_y = center
    r = radius * np.sqrt(np.random.random((number_of_points,)))
    theta = np.random.random((number_of_points,)) * 2 * np.pi
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    return x,y

oranges_x, oranges_y = points_within_circle(1.6, (5,2), 100)
lemons_x, lemons_y = points_within_circle(1.9, (2,5), 100)

class Perceptron(object):
    
    def __init__(self, weights, learning_rate=0.1):
        self.weights = np.array(weights)
        self.learning_rate = learning_rate

    @staticmethod
    def unit_step_function(x):        
        if x < 0:
            return 0
        else:
            return 1

    def __call__(self, in_data):
        weighted_input = self.weights * in_data
        weighted_sum = weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)
    
    def adjust(self,
               target_result,
               calculated_result,
               in_data):
        if type(in_data) != np.ndarray:
                in_data = np.array(in_data)
        error = target_result - calculated_result
        if error != 0:
            correction = error * in_data * self.learning_rate
            self.weights += correction
    
    def evaluate(self, data, labels):
        evaluation = Counter()
        for index in range(len(data)):
            label = int(round(self(data[index]), 0))

            if label == labels[index]:
                evaluation['corrects'] += 1
            else:
                evaluation['wrongs'] += 1
        return evaluation
    
p = Perceptron(weights = [0.1, 0.1], learning_rate=0.3)



print("training")

from sklearn.model_selection import train_test_split
import random

oranges = list(zip(oranges_x, oranges_y))
lemons = list(zip(lemons_x, lemons_y))

labelled_data = list(zip(oranges + lemons,
                         [0] * len(oranges) + [1] * len(lemons)))

random.shuffle(labelled_data)
data, labels = zip(*labelled_data)
res = train_test_split(data, labels, train_size=0.8,
                       test_size = 0.2,
                       random_state = 42)
train_data, test_data, train_labels, test_labels = res

for index in range(len(train_data)):
    p.adjust(train_labels[index],
             p(train_data[index]),
             train_data[index])


evaluation = p.evaluate(train_data, train_labels)
print(evaluation.most_common())
evaluation = p.evaluate(test_data, test_labels)
print(evaluation.most_common())

print(p.weights)
import matplotlib.pyplot as plt

X = np.arange(0, 7)
# fig, ax = plt.subplots()
# 
# lemons = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 1]
# lemons_x, lemons_y = zip(*lemons)
# oranges = [train_data[i] for i in range(len(train_data)) if train_labels[i] == 0]
# oranges_x, oranges_y = zip(*oranges)
# 
# ax.scatter(oranges_x, oranges_y, c="orange")
# ax.scatter(lemons_x, lemons_y, c="y")
# 
# w1 = p.weights[0]
# w2 = p.weights[1]
# m = -w1/w2
# ax.plot(X, m * X)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

p = Perceptron(weights=[0.1, 0.1],
               learning_rate=0.3)
number_of_colors = 7
colors = cm.rainbow(np.linspace(0, 1, number_of_colors))

fig, ax = plt.subplots()
ax.set_xticks(range(8))
ax.set_ylim([-2, 8])

counter = 0
for index in range(len(train_data)):
    old_weights = p.weights.copy()
    p.adjust(train_labels[index], 
             p(train_data[index]), 
             train_data[index])
    if not np.array_equal(old_weights, p.weights):
        color = "orange" if train_labels[index] == 0 else "y"        
        ax.scatter(train_data[index][0], 
                   train_data[index][1],
                   color=color)
        ax.annotate(str(counter), 
                    (train_data[index][0], train_data[index][1]))
        m = -p.weights[0] / p.weights[1]
        print(index, m, p.weights, train_data[index])
        ax.plot(X, m * X, label=str(counter), color=colors[counter])
        counter += 1
ax.legend()
plt.show()


