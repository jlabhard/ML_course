# Useful starting lines
import numpy as np
from implementations import *
from utilitary import *

from proj1_helpers import *

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

x1, x2, x3, x4, mask1, mask2, mask3, mask4 = split_categories(tX)
x1_test, x2_test, x3_test, x4_test, mask1_test, mask2_test, mask3_test, mask4_test = split_categories(tX_test)

''' Create a dictionary which enable to compute the different methods in a modularized way '''
methods = {
    "least_squares_GD" : (lambda y, tx, lambda_, initial_w, max_iters, gamma: least_squares_GD(y, tx, initial_w, max_iters, gamma)),
    "least_squares_SGD" : (lambda y, tx, lambda_, initial_w, max_iters, gamma: least_squares_SGD(y, tx, initial_w, max_iters, gamma)),
    "least_squares": (lambda y, tx, lambda_, initial_w, max_iters, gamma: least_squares(y, tx)),
    "ridge_regression" : (lambda y, tx, lambda_, initial_w, max_iters, gamma: ridge_regression(y, tx, lambda_)),
    "logistic_regression" : (lambda y, tx, lambda_, initial_w, max_iters, gamma: logistic_regression(y, tx, initial_w, max_iters, gamma)),
    "reg_logistic_regression" : (lambda y, tx, lambda_, initial_w, max_iters, gamma: reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma))
}

x1_normalized, x1_test_normalized = normalize(x1, x1_test)
x2_normalized, x2_test_normalized = normalize(x2, x2_test)
x3_normalized, x3_test_normalized = normalize(x3, x3_test)
x4_normalized, x4_test_normalized = normalize(x4, x4_test)

masks_test = np.array([mask1_test, mask2_test, mask3_test, mask4_test])

masks = np.array([mask1, mask2, mask3, mask4])
X = np.array([x1, x2, x3, x4])
X_normalized = np.array([x1_normalized, x2_normalized, x3_normalized, x4_normalized])

W, X_poly, optimal_degree, optimal_lambda, optimal_gamma, accuracy = analyze_method(methods["ridge_regression"], X_normalized, y, masks, degrees = np.arange(5, 11),lambdas = np.logspace(-18, -7, 12), name = "category ")

x1_test_poly = build_multi_poly(x1_test_normalized, optimal_degree[0])
x2_test_poly = build_multi_poly(x2_test_normalized, optimal_degree[1])
x3_test_poly = build_multi_poly(x3_test_normalized, optimal_degree[2])
x4_test_poly = build_multi_poly(x4_test_normalized, optimal_degree[3])

X_test = np.array([x1_test_poly, x2_test_poly, x3_test_poly, x4_test_poly])

OUTPUT_PATH = '../data/submission_file.csv' # TODO: fill in desired name of output file for submission
y_pred = np.zeros(tX_test.shape[0])

y_pred[masks_test[0]] = predict_labels(W[0], X_test[0])
y_pred[masks_test[1]] = predict_labels(W[1], X_test[1])
y_pred[masks_test[2]] = predict_labels(W[2], X_test[2])
y_pred[masks_test[3]] = predict_labels(W[3], X_test[3])
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
