# common accuracy class
import numpy as np


class Accuracy:

    # calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):
        # get comparison results
        comparisons = self.compare(predictions, y)  # implemented in the subclasses

        # calculate an accuracy
        accuracy = np.mean(comparisons)

        # return accuracy
        return accuracy


class RegressionAccuracy(Accuracy):

    def __init__(self):
        # create precision property
        self.precision = None

    # calculate precision value
    # based on passed in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # compares prediction to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
