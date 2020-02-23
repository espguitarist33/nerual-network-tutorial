'''
Created on Feb 23, 2020

@author: ryan
'''
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low-mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork(object):

    
    def create_weight_matricies(self):
        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(0, 1, -rad, rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes + bias_node))
        
        rad = 1 /np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(0, 1, -rad, rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes + bias_node))
    
    def train(self, input_vector, target_vector):
        bias_node = 1 if self.bias else 0
        if self.bias:
            #adding bias node to the end of the input_vector
            input_vector = np.concatenate((input_vector, [self.bias]))
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation_function(output_vector1)
        
        if self.bias:
            output_vector_hidden = np.concatenate((output_vector_hidden, [[self.bais]]))
        
        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp
        
        #calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1,:]
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_in_hidden += self.learning_rate * x
        
    
    def run(self, input_vector):
        """Run the netowrk with an input_vector tuple, list, or ndarray"""
        
        if self.bias:
            # add the node to the end of the input
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T
        
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)
        
        if self.bias:
            output_vector = np.concatenate((output_vector, [[1]]))
        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)
        
        return output_vector
    
    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 bias=None):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matricies()
        
# 
# simple_network = NeuralNetwork(no_of_in_nodes = 2,
#                                no_of_out_nodes=2,
#                                no_of_hidden_nodes=10,
#                                learning_rate=0.6)

# data1 = [((3, 4), (0.99, 0.01)), ((4.2, 5.3), (0.99, 0.01)), 
#          ((4, 3), (0.99, 0.01)), ((6, 5), (0.99, 0.01)), 
#          ((4, 6), (0.99, 0.01)), ((3.7, 5.8), (0.99, 0.01)), 
#          ((3.2, 4.6), (0.99, 0.01)), ((5.2, 5.9), (0.99, 0.01)), 
#          ((5, 4), (0.99, 0.01)), ((7, 4), (0.99, 0.01)), 
#          ((3, 7), (0.99, 0.01)), ((4.3, 4.3), (0.99, 0.01))]
# 
# data2 = [((-3, -4), (0.01, 0.99)), ((-2, -3.5), (0.01, 0.99)), 
#          ((-1, -6), (0.01, 0.99)), ((-3, -4.3), (0.01, 0.99)), 
#          ((-4, -5.6), (0.01, 0.99)), ((-3.2, -4.8), (0.01, 0.99)), 
#          ((-2.3, -4.3), (0.01, 0.99)), ((-2.7, -2.6), (0.01, 0.99)), 
#          ((-1.5, -3.6), (0.01, 0.99)), ((-3.6, -5.6), (0.01, 0.99)), 
#          ((-4.5, -4.6), (0.01, 0.99)), ((-3.7, -5.8), (0.01, 0.99))]
# 
# data = data1 + data2
# np.random.shuffle(data)
# 
# points1, labels1 = zip(*data1)
# X, Y = zip(*points1)
# plt.scatter(X, Y, c="r")
# 
# points2, labels2 = zip(*data2)
# X, Y = zip(*points2)
# plt.scatter(X, Y, c="b")
# 
# simple_network = NeuralNetwork(2, 2, 2, 0.6)
# 
# size_of_learning_sample = int(len(data)*0.9)
# learn_data = data[:size_of_learning_sample]
# test_data = data[-size_of_learning_sample:]
# 
# for i in range(size_of_learning_sample):
#     point, label = learn_data[i][0], learn_data[i][1]
#     simple_network.train(point, label)
# 
# for i in range(size_of_learning_sample):
#     point, label = learn_data[i][0], learn_data[i][1]
#     cls1, cls2 = simple_network.run(point)
#     print(point, cls1, cls2, end=": ")
#     if cls1 > cls2:
#         if label == (0.99, 0.01):
#             print("class1 correct", label)
#         else:
#             print("class2 incorrect", label)
#     else:
#         if label == (0.01, 0.99):
#             print("class 1 correct", label)
#         else:
#             print("class2 incorrect", label)

class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3) ] 
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6), 
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6), 
          (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8) ]

labeled_data = []
for el in class1:
    labeled_data.append([el, [1, 0]])
for el in class2:
    labeled_data.append([el, [0, 1]])

np.random.shuffle(labeled_data)
print(labeled_data[:10])

data, labels = zip(*labeled_data)
labels = np.array(labels)
data = np.array(data)

simple_network = NeuralNetwork(2,2,10,0.1,None)

for _ in range(20):
    for i in range(len(data)):
        simple_network.train(data[i], labels[i])
for i in range(len(data)):
    print(labels[i])
    print(simple_network.run(data[i]))


# plt.hist(s)
plt.show()


