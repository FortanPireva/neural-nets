def simplenn():
    inputs = [1, 2, 3, 3.5]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.2, 0.8, -0.5, 1],
               [0.2, 0.8, -0.5, 1],
               [0.2, 0.8, -0.5, 1]]
    biases = [2, 3, 0.5]

    layer_outputs = []

    for neuron_weights, neuron_bias in zip(weights, biases):

        neuron_output = 0

        for n_input, n_weight in zip(inputs, neuron_weights):
            neuron_output += n_input * n_weight

        neuron_output += neuron_bias

        layer_outputs.append(neuron_output)

    return layer_outputs
