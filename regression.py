import numpy as np

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data

from activation_relu import ReluActivation
from denser_layer import DenseLayer
from loss import MeanSquaredErrorLoss
from optimization.adamoptimizer import AdamOptimizer
from regression.activation_linear import LinearActivation

nnfs.init()

x, y = sine_data()

plt.plot(x, y)
plt.show()

# create dense layer with 1 input feature and 64 output values
dense1 = DenseLayer(1, 64)

# create relu activation ( to be used with dense layer)
activation1 = ReluActivation()

# create second dense layer with 64 input features and 64 output value

dense2 = DenseLayer(64, 64)

# create relu activation
activation2 = ReluActivation()

# create third dense layer with 64 input  features
# of previous layer and 1 output value

dense3 = DenseLayer(64, 1)

# craete linear activation
activation3 = LinearActivation()

# create loss function

loss_function = MeanSquaredErrorLoss()

# create optimizer

optimizer = AdamOptimizer(learning_rate=0.005, decay=1e-3)

# accuracy precision

accuracy_precision = np.std(y) / 250

# train in loop
for epoch in range(10001):

    # perform a forward pass of our training through first layer
    dense1.forward(x)

    # perform a forward pass through activation function
    activation1.forward(dense1.output)

    # perform forward pass through second dense layer
    dense2.forward(activation1.output)

    # perform forward pass trhough actvation function
    activation2.forward(dense2.output)

    # perform a forward pass through third dense layer
    # takes outputs of activation function of second layer as inputs
    dense3.forward(activation2.output)

    # perform forward pass through activation function
    # takes the output of third dense layer here
    activation3.forward(dense3.output)

    # calculate data loss
    data_loss = loss_function.calculate(activation3.output, y)

    # calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) + \
                          loss_function.regularization_loss(dense3)

    # calculate overall loss

    loss = data_loss + regularization_loss
    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value

    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # backward propagation
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update optimizer and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

import matplotlib.pyplot as plt

x_test, y_test = sine_data()

dense1.forward(x_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)

activation3.forward(dense3.output)

plt.plot(x_test, y_test)
plt.plot(x_test, activation3.output)
plt.show()
