import numpy as np


class ReluActivation:

    # forward pass
    def __init__(self):
        self.dinputs = None
        self.inputs = None
        self.output = None

    def forward(self, inputs, training):
        # remember input values
        self.inputs = inputs
        # calculate output values from input
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        # zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # calculate predictions
    def predictions(self, outputs):
            return outputs
