import numpy as np


class SigmoidActivation:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs

        self.output = 1 / (1 + np.exp(-self.inputs))

    # backward pass
    def backward(self, dvalues):
        # derivative - calculates output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output
