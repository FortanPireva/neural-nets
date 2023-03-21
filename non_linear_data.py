import nnfs
import numpy as np
from nnfs.datasets import spiral_data

from activation_relu import ReluActivation
from activation_softmax_loss_categorical_crossentropy import ActivationSoftmaxLossCategoricalCrossEntropy
from denser_layer import DenseLayer
from loss import CategoricalCrossEntropyLoss
from softmax import SoftmaxActivation

nnfs.init()

x,y = spiral_data(samples=100,classes=3)

# plt.scatter(x[:,0],x[:,1],c=y,cmap="brg")
# plt.show()


# create dense layer with 2 input features and 3 output values
dense1 = DenseLayer(2, 3)
# create ReLU activation ( to be used with Dense Layer)
activation1 = ReluActivation()

# create second dense layer with 3 input features(output of previous layer) and 3 output values
dense2 = DenseLayer(3, 3)

# create softmax classifier's combined loss and activation
loss_activation = ActivationSoftmaxLossCategoricalCrossEntropy()

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
loss = loss_activation.forward(dense2.output,y)

print(loss_activation.output[:5])


print('loss:', loss)

# calculate accuracy from output of activation2 and targets
# calculate valeus along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print('acc: ', accuracy)

# backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)