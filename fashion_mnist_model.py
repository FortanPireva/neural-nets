from zipfile import ZipFile
import os
import urllib
import urllib.request

from accuracy import CategoricalAccuracy
from activations.activation_relu import ReluActivation
from denser_layer import DenseLayer
from loss import CategoricalCrossEntropyLoss
from model import Model
from optimization.adamoptimizer import AdamOptimizer
from real_dataset.create_data_mnist import create_mnist_dataset
from softmax import SoftmaxActivation

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)

    print('Unzipping images...')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)

    print('DONE')

import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=250)

image_data = cv2.imread('fashion_mnist_images/train/4/0011.png',
                        cv2.IMREAD_UNCHANGED)

plt.imshow(image_data, cmap='gray') # tell matplotlib image is gray value
plt.show()


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


# instantiate the model
model = Model()

# add layers
model.add(DenseLayer(x.shape[1], 128))
model.add(ReluActivation())
model.add(DenseLayer(128, 128))
model.add(ReluActivation())
model.add(DenseLayer(128, 10))
model.add(SoftmaxActivation())

# set loss, optimizer and accuracy objects

model.set(
    loss = CategoricalCrossEntropyLoss(),
    optimizer=AdamOptimizer(decay=1e-3),
    accuracy=CategoricalAccuracy()
)

# finalize the model
model.finalize()
# train the model
# model.train(x, y, validation_data=(x_test, y_test), epochs=10, batch_size=128, print_every=100)

# load the parameters
model.load_parameters('fashion_mnist.params')
# evaluate the model
model.evaluate(x_test, y_test)

model.save('fashion_mnist.model')

# save parameters
# model.save_parameters('fashion_mnist.params')