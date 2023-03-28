import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2 =0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.output = None
        self.inputs = None
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.dinputs = None
        self.dbiases = None
        self.dweights = None
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        # remember input values
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        pass

    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # gradients on regularization
        # l1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0 ] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # l2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # l1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0 ] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # l2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.weight_regularizer_l2 * self.biases

        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases

    # set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases  = biases