# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

# Calculate the loss using MSE
def compute_loss(y, tx, w):

    # error
    e = y - np.sum(tx * w, axis=1)

    # error squared
    e2 = e * e

    # return the MSE
    return np.sum(e2) / (2 * len(y))

# Compute the gradient
def compute_gradient(y, tx, w):

    # Error
    e = y - np.sum(tx*w, axis = 1)

    # return the gradient
    return - tx.T.dot(e) / len(y)

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - np.sum(tx * w, axis=1)
    e2 = e * e

    return np.sum(e2) / (2 * len(y))

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.sum(tx*w, axis = 1)
    return - tx.T.dot(e) / len(y)

def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    # ws = [initial_w]
    # losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        # store w and loss
        # ws.append(w)
        # losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma) :
    # TODO: Implementation
    return NotImplementedError

def least_squares(y, tx) :
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_loss_mse(y, tx, w))

def ridge_regression(y, tx, lambda_) :
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])

    w = np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    loss = compute_rmse(y, tx, w)
    return w, loss

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
