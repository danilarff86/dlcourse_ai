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
        max_vals = max_vals[:,np.newaxis]
    norm = predictions - max_vals
    exps = np.exp(norm)
    sums = np.sum(exps, axis=-1)
    if predictions.ndim > 1:
        sums = np.repeat(sums[:,np.newaxis], predictions.shape[-1], axis=-1)
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
            np.array([np.arange(t_index_1d.shape[0]),t_index_1d], dtype=np.int),
            probs.shape)
        loss_arr = -np.log(np.ravel(probs)[flat_index_array])
        return np.mean(loss_arr)
    return -np.log(probs[target_index])

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    #raise Exception("Not implemented!")
    loss = np.sum(W**2) * reg_strength
    grad = W * (2* reg_strength)

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    #raise Exception("Not implemented!")
    probs = softmax(preds)
    loss_val = cross_entropy_loss(probs, target_index)

    ground_trues = np.zeros(preds.shape, dtype=np.float32)
    if preds.ndim > 1:
        t_index_1d = target_index.reshape(-1)
        flat_index_array = np.ravel_multi_index(
            np.array([np.arange(t_index_1d.shape[0]),t_index_1d], dtype=np.int),
            ground_trues.shape)
        np.ravel(ground_trues)[flat_index_array] = 1.0
    else:
        ground_trues[target_index] = 1.0

    dprediction = probs - ground_trues
    if preds.ndim > 1:
        dprediction /= dprediction.shape[0]
    loss = loss_val if preds.ndim == 1 else np.mean(loss_val)

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        #raise Exception("Not implemented!")
        self.X = X
        res = X.copy()
        res[res < 0] = 0
        return res

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        d_result = d_out.copy()
        d_result[self.X < 0] = 0
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        #raise Exception("Not implemented!")
        d_input = np.dot(d_out, self.W.value.T)
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.dot(np.ones((1,d_out.shape[0])), d_out)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
