'''
Created on Feb 22, 2020

@author: ryan
'''
import numpy as np

class Perceptron(object):
    def __init__(self, input_length, weights=None):
        if weights is None:
            self.weights = np.ones(input_length) / input_length
        else:
            self.weights = weights
            
    @staticmethod
    def unit_step_function(x):
        if x< 0:
            return 0
        return x
    
    def __call__(self, in_data):
        weighted_input = self.weights * in_data
        weighted_sum = weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)
    
p = Perceptron(2, np.array([0.45, 0.5]))

data_in = np.empty((2,))
for in1 in range(2):
    for in2 in range(2):
        data_in = (in1, in2)
        data_out = p(data_in)
        print(data_in, "---> ", data_out)