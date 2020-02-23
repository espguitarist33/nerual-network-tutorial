'''
Created on Feb 23, 2020

@author: ryan
'''
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter


class Perceptron:
    
    def __init__(self, input_length, weights=None, learning_rate=0.1):
        if weights is None:
            self.weights = np.ones(input_length) / input_length
        else:
            self.weights = weights
        self.learning_rate = learning_rate
        
    @staticmethod
    def unit_step_function(x):
        if x < 0:
            return 0
        return x

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
    
p = Perceptron(2, np.array([0.45, 0.5]))

# print("Network with binary")
# data_in = np.empty((2,))
# for in1 in range(2):
#     for in2 in range(2):
#         data_in = (in1, in2)
#         data_out = p(data_in)
#         print(data_in, "---> ", data_out)
# 
# print("Network with floats")
# data = [(0.4, 0.4), (0.6, 0.8)]
# for t in data:
#     data_out = p(t)
#     print(t, "---> ,", data_out)

from sklearn.datasets import make_blobs

fig, ax = plt.subplots()
# data, labels = make_blobs(n_samples=1000,
#                           centers = np.array([[2,2], [6,6]]),
#                           random_state=1)
# 
# for i in range(len(data)):
#     if labels[i] == 0:
#         ax.scatter(data[0], data[1], c="green")
#     else:
#         ax.scatter(data[0], data[1], c="blue")

# class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
#           (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3) ] 
# class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6), 
#           (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6), 
#           (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8) ]
# 
# X, Y = zip(*class1)
# plt.scatter(X,Y, c="r")
# 
# X, Y = zip(*class2)
# plt.scatter(X, Y, c="b")
# 
# p = Perceptron(2)
# 
# def lin1(x):
#     return x+4
# 
# for point in class1:
#     p.adjust(1,
#              p(point),
#              point)
# 
# for point in class2:
#     p.adjust(0,
#              p(point),
#              point)
#     
# evaluation = Counter()
# for point in chain(class1, class2):
#     if p(point) == 1:
#         evaluation['correct'] += 1
#     else:
#         evaluation["wrong"] +=1
# 
# testpoints = [(3.9, 6.9), (-2.9, -5.9)]
# for point in testpoints:
#     print(p(point))
#     
# 
# x = np.arange(-7, 10)
# y = 5*x + 10
# 
# m = -p.weights[0]/p.weights[1]
# plt.plot(x, m*x)

plt.show()



