from accuracy import RegressionAccuracy
from activations.activation_relu import ReluActivation
from denser_layer import DenseLayer
from dropout.DropoutLayer import DropoutLayer
from loss import MeanSquaredErrorLoss
from model import Model
from optimization.adamoptimizer import AdamOptimizer
from regression.activation_linear import LinearActivation
from nnfs.datasets import sine_data

# create dataset
x, y = sine_data();

# instantiate the model
model = Model()


# add layers
model.add(DenseLayer(1, 64))
model.add(ReluActivation())
model.add(DenseLayer(64, 64))
model.add(ReluActivation())
model.add(DenseLayer(64, 1))
model.add(LinearActivation())

# set loss function and optimizer

model.set(
    loss=MeanSquaredErrorLoss(),
    optimizer=AdamOptimizer(learning_rate=0.005, decay=1e-3),
    accuracy= RegressionAccuracy()
)

# finalize the model

model.finalize()

#  train the model
model.train(x, y, epochs=10000, print_every=100)

