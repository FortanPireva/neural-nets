import numpy as np

# passed in gradient from next layer
# use a vector of 1s

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# 3 sets of weights - one set for each neuron
# we  have 4 inputs - 4 weights
# keep weight transposed

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T


# # sum weights of given input
# # and multiply by the passed gradient for this neuron
# dx0 = sum(weights[0] * dvalues[0])
# dx1 = sum(weights[1] * dvalues[0])
# dx2 = sum(weights[2] * dvalues[0])
# dx3 = sum(weights[3] * dvalues[0])
#
# dinputs = np.array([dx0, dx1, dx2, dx3]) # gradient of the neuron function with respect to inputs
# print(dinputs)

# more simply
# sum weights of given input
# and multiply by the passed in gradient for this neuron

dinputs = np.dot(dvalues, weights.T)
print(dinputs)

# 3 sets iof inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# sum weights of given input
# and multiply by the passed in gradient for this neuron

dweights = np.dot(inputs.T, dvalues) # gradient of the neuron function with respect to the weights
print(dweights)

# one bias for each neuron
# biases are the row vector with shape (1,neurons)
biases = np.array([[2,3,0.5]])

dbiases = np.sum(dvalues,axis=0,keepdims=True) # keepdims lets us keep the gradient as row vector
# same as the shape of bias vector

# update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)