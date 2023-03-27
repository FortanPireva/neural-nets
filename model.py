# model class
from activations.activation_softmax_loss_categorical_crossentropy import ActivationSoftmaxLossCategoricalCrossEntropy
from input_layer import InputLayer
from loss import CategoricalCrossEntropyLoss
from softmax import SoftmaxActivation


class Model:

    def __init__(self):
        # create a list of network objects
        self.softmax_classsifier_output = None
        self.input_layer = None
        self.layers = []

        # loss function of this model
        self.loss = None

        # optimizer of the model
        self.optimizer = None

        # trainable layers
        self.trainable_layers = []

        # output layer activation
        self.output_layer_activation = None

        # accuracy
        self.accuracy = None

    # add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # use of * notes that the subsequent parameters are keyword arguments
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # train the model
    def train(self, x, y, *, epochs=1, print_every=1,
              validation_data = None):

        # initialize accuracy object
        self.accuracy.init(y)

        # main training loop
        for epoch in range(1, epochs + 1):
            # perform the forward pass
            output = self.forward(x, training=True)

            # calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            # get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # do the backpropagation
            self.backward(output, y)

            # optimize weights and biases of trainable_layers
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)

            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(
                    f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate}')
        if validation_data is not None:

            x_test, y_test = validation_data

            # perform forward pass
            output = self.forward(x_test,training=False)

            # calculate loss
            loss = self.loss.calculate(output, y_test)

            # get predictions anc calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_test)

            # print summary
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}' +
                  f'loss: {loss:.3f}'
                  )
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
                self.output_layer_activation = self.layers[i]

            # if layer contains an attribute called "weights"
            # it's a trainable layer
            # add it to list of trainable layers
            # we don't need to check for biases
            # checking for weights is enough

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        # update loss with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # if output activation is softmax and
        # loss function is categorical cross-entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], SoftmaxActivation) and \
            isinstance(self.loss, CategoricalCrossEntropyLoss):
            
            # create an object of combined activation
            # and loss function
            self.softmax_classsifier_output = ActivationSoftmaxLossCategoricalCrossEntropy()
            

    # forwrd pass
    def forward(self, x, training):

        # call forward method on input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(x, training)

        # call forward method of every object in chain
        # pass output of the previous object as parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # layer is now the last object from the list
        # return its output
        return layer.output

    # backward pass of the model class
    def backward(self, output, y):

        # if softmax classifier
        if self.softmax_classsifier_output is not None:
            # call backward on this object
            # will set the dinputs property
            self.softmax_classsifier_output.backward(output, y)

            # since we'll not call backward metjod of last layer
            # which is softmax activation
            # as we used combined softmax/loss
            # set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classsifier_output.dinputs

            # call backward method
            # of every layer going in reverse
            # except the last one
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        # first call backward method on the loss
        # that will set dinputs property on the last
        # layer will try to access
        self.loss.backward(output, y)

        # call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)