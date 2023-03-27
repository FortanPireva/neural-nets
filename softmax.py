import numpy as np


class SoftmaxActivation:
    def __init__(self):
        self.dinputs = None
        self.output = None

    def forward(self, inputs, training):
        # get normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # normalize them for each sampole
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        # create uniinitialized array
        self.dinputs = np.empty_like(dvalues)
        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1, 1)
            # calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        # calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
