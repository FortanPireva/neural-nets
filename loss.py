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
        
