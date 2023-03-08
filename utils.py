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


def softmax(inputs):
    exp_values = []
    for i in inputs:
        exp_values.append(math.pow(math.e, i))
    norm_value = sum(exp_values)
    output = []
    for i in inputs:
        output.append(i / norm_value)

    print("Normalized value", output)
    print("Sum of normalized values is ", sum(norm_value))
