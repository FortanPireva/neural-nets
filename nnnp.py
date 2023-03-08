import numpy as np


def simplenn():
    inputs = [1, 2, 3, 3.5]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.2, 0.8, -0.5, 1],
               [0.2, 0.8, -0.5, 1],
               [0.2, 0.8, -0.5, 1]]
    biases = [2, 3, 0.5,1]

    return np.dot(weights, inputs) + biases
