# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

''' Calculate the loss using MSE'''
def compute_mse_loss(y, tx, w):
    # error
    e = y - tx@w
    # return the loss
    return (e.T@e)/(2*len(y))

''' Compute the gradient of the mse loss'''
def compute_least_square_gradient(y, tx, w):
    # error
     e = y - tx@w
     # return the gradient
    return (-tx.T@e)/len(y)

"""Gradient descent algorithm using mse loss """
def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_least_square_gradient(y, tx, w)
        loss = compute_mse_loss(y, tx, w)
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

"""Stochastic gradient descent algorithm using mse loss """
def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        n = np.randint(N)
        batch_y = y[n]
        batch_x = x[n]
        grad = compute_least_square_gradient(batch_y, batch_tx, w)
        loss = compute_mse_loss(y, tx, w)
        w = w - gamma*grad
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

"""calculate the least squares solution using normal equation """
def least_squares(y, tx) :
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_mse_loss(y, tx, w)
    return w, loss

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse_loss(y, tx, w))

"""implement ridge regression."""
def ridge_regression(y, tx, lambda_) :
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])

    w = np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    loss = compute_rmse(y, tx, w)
    return w, loss

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1+np.exp(t))

def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1 + np.exp(tx@w))-y*tx@w)

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w) - y)

def logistic_regression(y, tx, initial_w, max_iters, gamma) :
    """
    Gradient descent algorithm.
    """
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad
    return w, loss

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = compute_logistic_loss(y, tx, w) + lambda_ * w.T.dot(w)
    gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma) :
    """
    Gradient descent algorithm.
    """
    w = initial_w
    for n_iter in range(max_iters):
        loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w -= gamma * gradient
    return w, loss

DATA_TEST_PATH = '' # TODO: download train data and supply path here
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
