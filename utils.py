import math


def relu(inputs):
    output = []
    for i in inputs:
        output.append(max(0, i))
    return output


def sigmoid(inputs):
    output = []
    for i in inputs:
        output.append(1 / (1 + math.pow(math.e, -i)))
    return output



