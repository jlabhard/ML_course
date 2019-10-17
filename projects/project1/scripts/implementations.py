# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *
DATA_TRAIN_PATH = '' # TODO: download train data and supply path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    # TODO: Implementation
    return NotImplementedError

def compute_loss(y, tx, w):
    """Calculate the loss MSE """
    e = y - tx@w
    return (e.T@e)/(2*len(y))

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx@w
    return (-1/len(y))*(tx.T@e)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size):
            grad = compute_stoch_gradient(batch_y, batch_tx, w)
            loss = compute_loss(y, tx, w)
            w = w - gamma*grad
    return w, loss

def least_squares(y, tx) :
    # TODO: Implementation
    return NotImplementedError

def ridge_regression(y, tx, lambda_) :
    # TODO: Implementation
    return NotImplementedError

def logistic_regression(y, tx, initial_w, max_iters, gamma) :
    # TODO: Implementation
    return NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma) :
    # TODO: Implementation
    return NotImplementedError

DATA_TEST_PATH = '' # TODO: download train data and supply path here
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
