import numpy as np


class Loss:

    # calculate the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # return loss
        return data_loss

    # regularization loss calculation
    def regularization_loss(self, layer):

        # 0 by default
        regularization_loss = 0

        # check for L1 regularization
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        # check for L2 regularization
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss


class CategoricalCrossEntropyLoss(Loss):

    # forward pass
    def __init__(self):
        self.dinputs = None

    def forward(self, y_pred, y_true):

        # number of samples in batch
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # losses
        negative_loss_likelihoods = -np.log(correct_confidences)
        return negative_loss_likelihoods

    # backward pass
    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues

        # normalize gradient
        self.dinputs = self.dinputs / samples


# binary cross-entropy loss
class BinaryCrossEntropyLoss(Loss):

    def __init__(self):
        self.dinputs = None

    # forward pass
    def forward(self, y_pred, y_true):
        # clip data to prevent division by zero
        # clip both sides to not drag mean toward any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # calculate sample-wise loss
        sample_losses = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        sample_losses = np.mean(sample_losses, axis=-1)

        # return sample_losses
        return sample_losses

    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        # number of outputs in every sample
        outputs = len(dvalues[0])

        # clip data to prevent division by 0
        # clip both sides so to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

        # normalize gradient
        self.dinputs = self.dinputs / samples



class MeanSquaredErrorLoss(Loss):


    def __init__(self):
        self.dinputs = None
        self.output = None

    # forward pass
    def forward(self, y_pred, y_true):

        # calculate loss
        sample_losses = np.mean((y_true - y_pred) **2, axis=-1)
        return sample_losses


    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        # number of outputs
        outputs = len(dvalues[0])

        # gradient on values
        self.dinputs = dvalues * 2 * (y_true - dvalues) / outputs
        # normalize gradient
        self.dinputs = self.dinputs / samples



class MeanAbsoluteError(Loss):

    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None
    #forward pass
    def forward(self, y_pred, y_true):

        # calculate loss
        sample_loss = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_loss

    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        # number of outputs
        outputs = len(samples[0])


        # calculate gradients

        self.dinputs = np.sign(y_true - dvalues) / outputs

        # normalize gradients
        self.dinputs = self.dinputs / samples


