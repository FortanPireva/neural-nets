import numpy as np


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        # if layer does not contain cache arrays
        # create them filled with zeros
        if not hasattr(layer, "weights_cache"):
            layer.weights_momentums = np.zeros_like(layer.weights)
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_momentums = np.zeros_like(layer.biases)
            layer.biases_cache = np.zeros_like(layer.biases)

        layer.weights_momentums = self.beta_1 * \
                                  layer.weights_momentums + \
                                  (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.biases_momentums + \
                               (1 - self.beta_1) * layer.dbiases

        # get corrected momentum
        # self.iterations is 0 for first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weights_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))

        # update cache
        # with squared current gradients
        layer.weights_cache = self.beta_2 * layer.weights_cache + \
                              (1 - self.beta_2) * layer.dweights ** 2
        layer.biases_cache = self.beta_2 * layer.biases_cache + \
                             (1 - self.beta_2) * layer.dbiases ** 2

        # get corrected cache
        weights_cache_corrected = layer.weights_cache / \
                                  (1 - self.beta_2 ** (self.iterations + 1))
        biases_cache_corrected = layer.biases_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))

        # vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weights_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(biases_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
