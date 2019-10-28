# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from implementation_helper import *

"""Gradient descent algorithm using mse loss """
def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_least_square_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient
    return w, loss

"""Stochastic gradient descent algorithm using mse loss """
def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, 1):
            grad = compute_least_square_gradient(batch_y, batch_tx, w)
            loss = compute_mse(y, tx, w)
            w = w - gamma*grad
    return w, loss

"""calculate the least squares solution using normal equation """
def least_squares(y, tx) :
    try :
        w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    except :
        w = np.linalg.lstsq(tx.T @ tx, tx.T @ y, rcond = -1)[0]
    loss = compute_mse(y, tx, w)
    return w, loss


"""implement ridge regression."""
def ridge_regression(y, tx, lambda_) :
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])

    try :
        w = np.linalg.solve(tx.T @ tx + aI, tx.T @ y)
    except :
        left = (tx.T @ tx + aI)
        right = tx.T @ y
        w = np.linalg.lstsq(left, right, rcond = -1 )[0]
    loss = compute_mse(y, tx, w)
    return w, loss

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

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma) :
    """
    Gradient descent algorithm.
    """
    w = initial_w
    for n_iter in range(max_iters):
        loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w -= gamma * gradient
    return w, loss
