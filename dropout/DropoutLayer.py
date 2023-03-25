import numpy as np


class DropoutLayer:

    # init
    def __init__(self, rate):
        self.success_rate = 1 - rate
        self.inputs = None
        self.output = None
        self.dinputs = None

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs

        # generate binary mask
        self.binary_mask = np.random.binomial(1, self.success_rate, size=self.inputs.shape) / self.success_rate

        # apply mask to output values

        self.output = self.inputs * self.binary_mask

    def backward(self, dvalues):
        # gradients on values

        self.dinputs = dvalues * self.binary_mask
