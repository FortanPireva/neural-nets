import numpy as np
def batch_np():
    inputs = [[1, 2, 3, 3.5],[3,3,3,3],[2,2,2,2],[1,1,1,1]]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.2, 0.8, -0.5, 1],
               [0.2, 0.8, -0.5, 1],
               [0.2, 0.8, -0.5, 1]]
    biases = [2, 3, 0.5, 1]

    layer_outputs = np.dot(inputs,np.array(weights).T) + biases

    print(layer_outputs)