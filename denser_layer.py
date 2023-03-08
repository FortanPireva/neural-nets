import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        pass

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass
