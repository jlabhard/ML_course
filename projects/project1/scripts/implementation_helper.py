# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_loss_mse(y, tx, w))

''' Calculate the loss using MSE'''
def compute_mse(y, tx, w):
    # error
    e = y - tx @ w
    # return the loss
    return (e.T @ e)/(2*len(y))

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y, tx, w))

''' Compute the gradient of the mse loss'''
def compute_least_square_gradient(y, tx, w):
    # error
    e = y - tx @ w
    # return the gradient
    return (-tx.T @ e)/len(y)

"""Stochastic gradient descent algorithm using mse loss """
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.where(t >= 0,
                    1 / (1 + np.exp(-t)),
                    np.exp(t) / (1 + np.exp(t)))

def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    y = (y+1)/2
    return np.sum(np.log(1 + np.exp(tx @ w)) - y * (tx @ w))

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    y = (y+1)/2
    return tx.T @ (sigmoid(tx @ w) - y)

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = compute_logistic_loss(y, tx, w) + (lambda_/2) * (w.T @ w)
    gradient = compute_logistic_gradient(y, tx, w) + lambda_ * w
    return loss, gradient
