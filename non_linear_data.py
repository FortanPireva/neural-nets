import nnfs
import numpy as np
from matplotlib import pyplot as plt
from nnfs.datasets import spiral_data

from activation_relu import ReluActivation
from activation_softmax_loss_categorical_crossentropy import ActivationSoftmaxLossCategoricalCrossEntropy
from denser_layer import DenseLayer
from loss import CategoricalCrossEntropyLoss
from optimization.RMSPropOptimizer import RMSPropOptimizer
from optimization.adagradoptimizer import AdaGradOptimizer
from optimization.adamoptimizer import AdamOptimizer
from optimization.sgdoptimizer import SGDOptimizer
from softmax import SoftmaxActivation

nnfs.init()

x, y = spiral_data(samples=100, classes=3)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap="brg")
plt.show()

# create dense layer with 2 input features and 64 output values
dense1 = DenseLayer(2, 64, weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
# create ReLU activation ( to be used with Dense Layer)
activation1 = ReluActivation()

# create second dense layer with 64 input features(output of previous layer) and 3 output values
dense2 = DenseLayer(64, 3)

# create softmax classifier's combined loss and activation
loss_activation = ActivationSoftmaxLossCategoricalCrossEntropy()

# create optimizer
optimizer = AdamOptimizer(learning_rate=0.03, decay=1e-5)  # 0.00

# train in loop
for epoch in range(10001):
    # perform a forward pass of our training data through this layer
    dense1.forward(x)

    # perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # perform a forward pass through second Dense Layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # perform forward through the activation/loss function
    # takes output of second dense layer and returns loss
    data_loss = loss_activation.forward(dense2.output, y)

    # calculate regularization penalty
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
                          loss_activation.loss.regularization_loss(dense2)

    # calculate overall loss
    loss = data_loss + regularization_loss

    # calculate accuracy from output of activation2 and targets
    # calculate valeus along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch} ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} ' +
              f'data_loss : {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}, '
              f'lr: {optimizer.current_learning_rate}')

    # backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# validate the model

# create test dataset
x_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(x_test)

activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
