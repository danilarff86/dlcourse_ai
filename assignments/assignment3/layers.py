import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    max_vals = np.max(predictions, axis=-1)
    if predictions.ndim > 1:
        max_vals = max_vals[:, np.newaxis]
    norm = predictions - max_vals
    exps = np.exp(norm)
    sums = np.sum(exps, axis=-1)
    if predictions.ndim > 1:
        sums = np.repeat(sums[:, np.newaxis], predictions.shape[-1], axis=-1)
    return exps / sums


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    if probs.ndim > 1:
        t_index_1d = target_index.reshape(-1)
        flat_index_array = np.ravel_multi_index(
            np.array([np.arange(t_index_1d.shape[0]),
                      t_index_1d], dtype=np.int),
            probs.shape)
        loss_arr = -np.log(np.ravel(probs)[flat_index_array])
        return np.mean(loss_arr)
    return -np.log(probs[target_index])


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    #raise Exception("Not implemented!")
    loss = np.sum(W**2) * reg_strength
    grad = W * (2* reg_strength)

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    #raise Exception("Not implemented!")
    probs = softmax(predictions)
    loss_val = cross_entropy_loss(probs, target_index)

    ground_trues = np.zeros(predictions.shape, dtype=np.float32)
    if predictions.ndim > 1:
        t_index_1d = target_index.reshape(-1)
        flat_index_array = np.ravel_multi_index(
            np.array([np.arange(t_index_1d.shape[0]),
                      t_index_1d], dtype=np.int),
            ground_trues.shape)
        np.ravel(ground_trues)[flat_index_array] = 1.0
    else:
        ground_trues[target_index] = 1.0

    dprediction = probs - ground_trues
    if predictions.ndim > 1:
        dprediction /= dprediction.shape[0]
    loss = loss_val if predictions.ndim == 1 else np.mean(loss_val)

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        self.X = X
        res = X.copy()
        res[res < 0] = 0
        return res

    def backward(self, d_out):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        d_result = d_out.copy()
        d_result[self.X < 0] = 0
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment

        #raise Exception("Not implemented!")
        d_input = np.dot(d_out, self.W.value.T)
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.dot(np.ones((1,d_out.shape[0])), d_out)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        batch_size, height, width, channels = X.shape
        self.X = np.pad(X, (
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
            (0, 0)
        ), 'constant')

        out_height = self.X.shape[1] - self.filter_size + 1
        out_width = self.X.shape[2] - self.filter_size + 1

        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        W_flatten = self.W.value.reshape((-1, self.out_channels))

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                # pass
                mini_X = self.X[:, y:y+self.filter_size, x:x +
                                self.filter_size, :].reshape((batch_size, -1))
                new_value = mini_X.dot(W_flatten) + self.B.value
                out[:, y, x, :] = new_value
        #raise Exception("Not implemented!")
        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_input = np.zeros(self.X.shape)

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                #pass
                mini_X = self.X[:, y:y+self.filter_size, x:x +
                                self.filter_size, :].reshape((batch_size, -1))
                mini_d_out = d_out[:, y, x, :].reshape((batch_size, out_channels))

                self.W.grad += np.dot(mini_X.T, mini_d_out).reshape(self.W.grad.shape)
                self.B.grad += mini_d_out.sum(axis=0)

                W_flatten = self.W.value.reshape((-1, self.out_channels))

                new_value = mini_d_out.dot(W_flatten.T).reshape(
                    (batch_size, self.filter_size, self.filter_size, channels))

                d_input[:, y:y+self.filter_size, x:x + self.filter_size, :] += new_value

        #raise Exception("Not implemented!")
        if self.padding > 0:
            d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding, :]
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.X_argmax_mask = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        #raise Exception("Not implemented!")
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        self.X = X
        out = np.zeros((batch_size, out_height, out_width, channels))
        self.X_argmax_mask = np.zeros_like(out)

        for y in range(out_height):
            for x in range(out_width):
                x_in = x * self.stride
                y_in = y * self.stride

                mini_X = self.X[:, y_in:y_in+self.pool_size, x_in:x_in +
                                self.pool_size, :].reshape((X.shape[0], -1, X.shape[3]))
                self.X_argmax_mask[:, y, x, :] = np.argmax(mini_X, axis=1)
                out[:, y, x, :] = np.max(mini_X, axis=1)
        
        self.X_argmax_mask = self.X_argmax_mask.astype(int)

        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        #raise Exception("Not implemented!")
        d_in = np.zeros_like(self.X)

        for y in range(d_out.shape[1]):
            for x in range(d_out.shape[2]):
                new_value = np.zeros((
                    self.X.shape[0],
                    self.pool_size ** 2,
                    self.X.shape[3]))

                for i in range(d_out.shape[0]):
                    for j in range(d_out.shape[3]):
                        new_value[i, self.X_argmax_mask[i, y, x, j],
                                  j] = d_out[i, y, x, j]

                x_in = x * self.stride
                y_in = y * self.stride

                d_in[:, y_in:y_in+self.pool_size, x_in:x_in + self.pool_size, :] = new_value.reshape((
                    self.X.shape[0],
                    self.pool_size,
                    self.pool_size,
                    self.X.shape[3]))
        return d_in

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        #raise Exception("Not implemented!")
        self.X_shape = X.shape
        return X.reshape((batch_size, height*width*channels))

    def backward(self, d_out):
        # TODO: Implement backward pass
        #raise Exception("Not implemented!")
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
