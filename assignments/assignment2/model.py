import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        for layer in self.layers:
            layer_params = layer.params()
            for param_key in layer_params:
                param = layer_params[param_key].grad = np.zeros(layer_params[param_key].value.shape)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        next_in = X
        for layer in self.layers:
            next_in = layer.forward(next_in)

        loss, grad = softmax_with_cross_entropy(next_in, y)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")
        for p in self.params().values():
            loss_reg, grad_reg = l2_regularization(p.value, self.reg)
            p.grad += grad_reg / p.value.shape[0]
            loss += loss_reg / p.value.shape[0]

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        #pred = np.zeros(X.shape[0], np.int)
        #raise Exception("Not implemented!")

        next_in = X
        for layer in self.layers:
            next_in = layer.forward(next_in)

        pred = softmax(next_in)

        return pred.argmax(axis=-1)

    def params(self):
        result = {}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_params = layer.params()
            for param in layer_params:
                result[param + str(i)] = layer_params[param]

        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")

        return result
