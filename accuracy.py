# common accuracy class
import numpy as np


class Accuracy:

    # calculates an accuracy
    # given predictions and ground truth values
    def __init__(self):
        self.accumulated_count = None
        self.accumulated_sum = None

    def calculate(self, predictions, y):
        # get comparison results
        comparisons = self.compare(predictions, y)  # implemented in the subclasses

        # calculate an accuracy
        accuracy = np.mean(comparisons)
        
        # add accumulkated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        # return accuracy
        return accuracy

    # calculate accumulated accuracy
    def calculate_accumulated(self):

        return self.accumulated_sum / self.accumulated_count

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
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


# accuracy calculation for classification Model
class CategoricalAccuracy(Accuracy):

    # no initialization is needed
    # still needs to exist as it will be called
    # from train method inside the model class
    def init(self, y):
        pass

    # compares predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


