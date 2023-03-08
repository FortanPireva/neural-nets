import numpy as np
from nnfs.datasets import spiral_data
import nnfs

from denser_layer import DenseLayer

nnfs.init()

import matplotlib.pyplot as plt

x,y = spiral_data(samples=100,classes=3)

# plt.scatter(x[:,0],x[:,1],c=y,cmap="brg")
# plt.show()


# create dense layer with 2 input features and 3 output values
dense1 = DenseLayer(2,3)
dense1.forward(x)

print(dense1.output[:5])



