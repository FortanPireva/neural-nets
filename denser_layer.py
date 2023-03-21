import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.inputs = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.dinputs = None
        self.dbiases = None
        self.dweights = None

    def forward(self, inputs):
        # remember input values
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        pass

    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
