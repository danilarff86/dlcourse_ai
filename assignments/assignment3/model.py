import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
        self.layers = [
            ConvolutionalLayer(
                in_channels=input_shape[2], out_channels=conv1_channels, filter_size=3, padding=0),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(
                in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=0),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(n_input=conv2_channels,
                                n_output=n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        #raise Exception("Not implemented!")
        for layer in self.layers:
            layer_params = layer.params()
            for param_key in layer_params:
                param = layer_params[param_key].grad = np.zeros(
                    layer_params[param_key].value.shape)

        # Forward pass
        next_in = X
        for layer in self.layers:
            next_in = layer.forward(next_in)

        loss, grad = softmax_with_cross_entropy(next_in, y)

        # Backward pass
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        # for p in self.params().values():
        #     loss_reg, grad_reg = l2_regularization(p.value, self.reg)
        #     p.grad += grad_reg / p.value.shape[0]
        #     loss += loss_reg / p.value.shape[0]

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        #raise Exception("Not implemented!")
        next_in = X
        for layer in self.layers:
            next_in = layer.forward(next_in)

        pred = softmax(next_in)

        return pred.argmax(axis=-1)

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        #raise Exception("Not implemented!")
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_params = layer.params()
            for param in layer_params:
                result[param + str(i)] = layer_params[param]

        return result
