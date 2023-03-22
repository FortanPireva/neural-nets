import numpy as np


class RMSPropOptimizer:
    def __init__(self, learning_rate = 0.001, decay = 0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho


    # pre-calculate update parameters
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))


    def update_params(self,layer):
        if not hasattr(layer,'weights_cache'):
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_cache = np.zeros_like(layer.biases)
        # update cache with squared current gradients
        layer.weights_cache = self.rho * layer.weights_cache + \
                              (1 - self.rho) * layer.dweights**2
        layer.biases_cache = self.rho * layer.biases_cache + \
                             (1 - self.rho) * layer.dbiases**2
        # vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                            layer.dweights / \
                            (np.sqrt(layer.weights_cache) + self.epsilon)

        layer.biases += -self.current_learning_rate * \
                            layer.dbiases / \
                            (np.sqrt(layer.biases_cache) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1