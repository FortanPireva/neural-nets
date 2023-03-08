import numpy as np


class ReluActivation:

    # forward pass
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # calculate output values from input
        self.output = np.maximum(0, inputs)
