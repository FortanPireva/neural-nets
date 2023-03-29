import cv2
import numpy as np

from model import Model
from real_dataset.create_data_mnist import create_mnist_dataset

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# read the image
image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)

# resize to the same size as fashion mnist images
image_data = cv2.resize(image_data, (28,28))

# invert image colors
image_data = 255 - image_data

# reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# load the model
model = Model.load('fashion_mnist.model')

#  predict  on the first 5 samples from validation dataset
# and print  the result
confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)

# get label from the label index
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)



