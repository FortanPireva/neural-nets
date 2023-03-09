import numpy as np
from nnfs.datasets import spiral_data
import nnfs

from activation_relu import ReluActivation
from denser_layer import DenseLayer
from loss import CategoricalCrossEntropyLoss
from softmax import SoftmaxActivation

nnfs.init()

import matplotlib.pyplot as plt

x,y = spiral_data(samples=100,classes=3)

# plt.scatter(x[:,0],x[:,1],c=y,cmap="brg")
# plt.show()


# create dense layer with 2 input features and 3 output values
dense1 = DenseLayer(2,3)
dense1.forward(x)

activation = ReluActivation()

activation.forward(dense1.output)

dense2 = DenseLayer(3,3)

dense2.forward(activation.output)

activation2 = SoftmaxActivation()

activation2.forward(dense2.output)

# create loss function
loss_function = CategoricalCrossEntropyLoss()


print(activation2.output[:5])

# perform a forward pass through loss functino
# it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output,y)

print('loss:', loss)

# calculate accuracy from output of activation2 and targets
# calculate valeus along first axis
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print('acc: ', accuracy)