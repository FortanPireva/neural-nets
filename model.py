# model class
from input_layer import InputLayer


class Model:

    def __init__(self):
        # create a list of network objects
        self.input_layer = None
        self.layers = []

        # loss function of this model
        self.loss = None

        # optimizer of the model
        self.optimizer = None

        # trainable layers
        self.trainable_layers = None

    # add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # use of * notes that the subsequent parameters are keyword arguments
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    # train the model
    def train(self, x, y, *, epochs=1, print_every=1):

        # main training loop
        for epoch in range(1, epochs + 1):
            # perform the forward pass
            output = self.forward(x)

            # temporary
            print(output)
            exit()

    # finalize the model
    def finalize(self):

        # create and set the input layer
        self.input_layer = InputLayer()

        # count all the objects
        layer_count = len(self.layers)

        # iterate the objects
        for i in range(layer_count):

            # if it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            # all layers except the first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss

            # if layer contains an attribute called "weights"
            # it's a trainable layer
            # add it to list of trainable layers
            # we don't need to check for biases
            # checking for weights is enough

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])


    # forwrd pass
    def forward(self, x):

        # call forward method on input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(x)

        # call forward method of every object in chain
        # pass output of the previous object as parameter
        for layer in self.layers:
            layer.forward(layer.prev.output)

        # layer is now the last object from the list
        # return its output
        return layer.output
