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

"""Gradient descent algorithm using mse loss """
def least_squares_GD(y, tx, initial_w, max_iters, gamma) :
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_least_square_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #   bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
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
        loss = compute_mse(y, tx, w)
        w = w - gamma*grad
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

"""calculate the least squares solution using normal equation """
def least_squares(y, tx) :
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_mse(y, tx, w)
    return w, loss


"""implement ridge regression."""
def ridge_regression(y, tx, lambda_) :
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])

    w = np.linalg.solve(tx.T @ tx + aI, tx.T @ y)
    loss = compute_mse(y, tx, w)
    return w, loss

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
    loss = compute_logistic_loss(y, tx, w) + (lambda_/2) * (w.T @ w)
    gradient = compute_logistic_gradient(y, tx, w) + lambda_ * w
    print("gradient : ", compute_logistic_gradient(y, tx, w))
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

def split_data(x, y, ratio, seed):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    temp = list(zip(x,y))
    np.random.shuffle(temp)
    x,y = zip(*temp)
    threshold = int(len(x)*ratio)
    x_split = np.split(x, [threshold, len(x)])
    y_split = np.split(y, [threshold, len(y)])
    
    return x_split[0], x_split[1], y_split[0], y_split[1]

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.zeros((degree+1, len(x)))
    for deg in range(degree+1):
        y = np.power(x,deg)
        poly[deg] = y
    return poly.T

def plot_train_test(train_errors, test_errors, x_axis, title_):
    """
    train_errors, test_errors should be list (of the same size) the respective train error and test error ,
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(x_axis, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(x_axis, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("degree")
    plt.ylabel("RMSE")
    plt.title(title_)
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.array([x,]*(degree+1)).transpose()
    powers = np.tile(range(degree+1), (len(x), 1))
    return phi**powers

def build_multi_poly(X, degree) :
    poly = np.ones(X.shape[0])
    for i in range(X.shape[1]) :
        feature = X[:, i]
        feature_poly = build_poly(feature, degree)[:, 1:]
        poly = np.c_[poly, feature_poly]
    return poly