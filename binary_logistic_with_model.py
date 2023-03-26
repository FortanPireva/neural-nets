from accuracy import RegressionAccuracy, CategoricalAccuracy
from activations.activation_relu import ReluActivation
from activations.activation_sigmoid import SigmoidActivation
from denser_layer import DenseLayer
from loss import MeanSquaredErrorLoss, BinaryCrossEntropyLoss
from model import Model
from optimization.adamoptimizer import AdamOptimizer
from regression.activation_linear import LinearActivation
from nnfs.datasets import sine_data
from nnfs.datasets import spiral_data
# create dataset
x, y = spiral_data(samples=100, classes=2)
x_test, y_test = spiral_data(samples=100, classes=2)

# reshape labels to list of lists
# inner list contain 1 ouput ( 0 or 1)
# per eeach output neuron, 1 in this case
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# instantiate the model
model = Model()

# add layers
model.add(DenseLayer(2,64, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4))
model.add(ReluActivation())
model.add(DenseLayer(64,1))
model.add(SigmoidActivation())

# set loss, optimizer ad accuracy objects

model.set(
    loss= BinaryCrossEntropyLoss(),
    optimizer=AdamOptimizer(decay=5e-7),
    accuracy=CategoricalAccuracy()
)

# finalize the model
model.finalize()
# train the model
model.train(x, y, epochs=10000, print_every=100, validation_data=(x_test, y_test))

