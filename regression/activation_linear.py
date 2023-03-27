class LinearActivation:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs, training):
        # linear function f(x) = x
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues
        self.dinputs = dvalues.copy()

    # calculate predictions
    def predictions(self, outputs):
        return outputs