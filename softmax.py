import numpy as np


class SoftmaxActivation:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # get normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # normalize them for each sampole
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
