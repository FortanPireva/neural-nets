import numpy as np
import cv2
import os


# loads mnist dataset
def load_mnist_dataset(dataset, path):
    # labels for each dataset in path
    labels = os.listdir(os.path.join(path, dataset))

    # create lists for samples and labels
    x = []
    y = []

    for label in labels:
        # get all files in given folder:
        for file in os.listdir(os.path.join(path, dataset, label)):
            # read the image
            image = cv2.imread(os.path.join(path, dataset, label, file),
                               cv2.IMREAD_UNCHANGED)

            # and append image as sample and label as y
            x.append(image)
            y.append(label)

    return np.array(x), np.array(y).astype('uint8')


def create_mnist_dataset(path):
    # load both sets separately
    x, y = load_mnist_dataset('train', path)
    x_test, y_test = load_mnist_dataset('test', path)

    return x, y, x_test, y_test
