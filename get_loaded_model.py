import numpy as np

from model import Model
from real_dataset.create_data_mnist import create_mnist_dataset

# create dataset

x, y, x_test, y_test = create_mnist_dataset('fashion_mnist_images')

# shuffle the data
keys = np.array(range(x.shape[0]))
np.random.shuffle(keys)

x = x[keys]
y = y[keys]

# then flatten sample-wise and scale in range -1 to 1
x = (x.reshape(x.shape[0], -1).astype(np.float32) - 127.5) / 127.5
x_test = (x_test.reshape(x_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5


# load the model
model = Model.load('fashion_mnist.model')

# predict on the first 5 samples from validation dataset
# and print the result
confidences = model.predict(x_test[:5])
print(confidences)