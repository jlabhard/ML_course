import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *


def split_data(x, y, ratio, seed):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
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

def plot_train_test(train_accuracy, test_accuracy, x_axis, title_, p, log = False):
    """
    train_errors, test_errors should be list (of the same size) the respective train error and test error ,
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    if log :
        plt.semilogx(x_axis, train_accuracy, color='b', marker='*', label="Train accuracy")
        plt.semilogx(x_axis, test_accuracy, color='r', marker='*', label="Test accuracy")
    else :
        plt.plot(x_axis, train_accuracy, color='b', marker='*', label="Train accuracy")
        plt.plot(x_axis, test_accuracy, color='r', marker='*', label="Test accuracy")
    plt.xlabel(p)
    plt.ylabel("Accuracy")
    plt.title(title_)
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.show()


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

def partition(number):
    answer = []
    answer.append((number, ))
    for x in range(1, number):
        for y in partition(number - x):
            answer.append((x,  ) + y)
    answer = [ans for ans in answer if len(ans) <3]
    return answer

def correct_partition(arr):
    return [ans for ans in arr if len(ans) != 1  ]

def other_partition(arr):
    arr = correct_partition(arr)
    return [ans for ans in arr if (ans[0] == 0 or ans[1] == 0) ]


# def build_augmented_features(feature_1, feature_2, degree=2, cross= True):
#     degree_array = []
#     if cross:
#         for i in range(degree):
#             degree_array.append(correct_partition(partition(i+1)))
#     else:
#         for i in range(degree):
#             degree_array.append(other_partition(partition(i+1)))

#     degree_array = [item for sublist in degree_array for item in sublist]
#     augmented_array = np.zeros((feature_1.shape[0], len(degree_array)))

#     for i, tuple_ in enumerate(degree_array):
#         augmented_array[:, i] = (feature_1 * tuple_[0]) * (feature_2 * tuple_[1])

#     return augmented_array

def build_augmented_features(feature_1, feature_2, degree=2, cross= True):
    degree_array = []
    if cross:
        for i in range(degree):
            degree_array.append(correct_partition(partition(i+1)))
    else:
        for i in range(degree):
            degree_array.append(other_partition(partition(i+1)))

    degree_array = [item for sublist in degree_array for item in sublist]

    augmented_feat_1 = np.tile(feature_1, (len(degree_array), 1)).T
    augmented_feat_2 = np.tile(feature_2, (len(degree_array), 1)).T
    degree_feat_1 = np.tile(np.array([item[0] for item in degree_array]), (feature_1.shape[0], 1))
    degree_feat_2 = np.tile(np.array([item[1] for item in degree_array]), (feature_2.shape[0], 1))

    augmented_array = (augmented_feat_1 * degree_feat_1) * (augmented_feat_2 * degree_feat_2)
    return augmented_array

def build_all(dataset, degree, cross = True):
    augs = []
    for pair in get_combinations(np.arange(dataset.shape[1])):
        aug = build_augmented_features(dataset[:, pair[0]], dataset[:, pair[1]], degree, cross)
        augs.append(aug)
    tX_standardized_af_aug = np.concatenate(augs, axis=1)
    tX_standardized_af_aug = np.c_[build_multi_poly(dataset, degree), tX_standardized_af_aug]
    return tX_standardized_af_aug

def get_combinations(arr):
    my_combinations = []
    for i in range(len(arr)):
        for j in range(len(arr)-1-i):
            my_combinations.append((arr[i], arr[i+j+1]))
    return my_combinations

def build_mask(tx) :
    mask1 = tx[:, 0] == -999
    mask2 = (tx[:, 23] == -999) & (tx[:, 0] != -999)
    mask3 = (tx[:, 4] == -999) & (tx[:, 23] != -999) & (tx[:, 0] != -999)
    mask4 = (tx[:, 4] != -999) & (tx[:, 23] != -999) & (tx[:, 0] != -999)

    return mask1, mask2, mask3, mask4

def build_subsample(tx, mask) :
    feature_mask = []
    for j in range(tx.shape[1]):
        if (j == 22) :
            feature_mask.append(not ((0 in tx[mask][:,j]) | (1 in tx[mask][:,j])))
        elif (j == 29) :
            feature_mask.append(not ((0 in tx[mask][:,j])))
        else :
            feature_mask.append(not (-999 in tx[mask][:,j]))

    subsample = tx[mask][:, feature_mask]

    return subsample

def split_categories(tx) :
    m1, m2, m3, m4  = build_mask(tx)

    s1 = build_subsample(tx, m1)
    s2 = build_subsample(tx, m2)
    s3 = build_subsample(tx, m3)
    s4 = build_subsample(tx, m4)

    return s1, s2, s3, s4, m1, m2, m3, m4

#normalize the matrix
def normalize(tx, tx_test) :
    tx_normalized = tx.copy()
    tx_test_normalized = tx_test.copy()
    for index, feature in enumerate(tx_normalized.T) :
        diff = np.amax(feature) - np.amin(feature)
        normalized_feature = (feature - np.amin(feature))/diff
        normalized_test_feature = (tx_test_normalized.T[index] - np.amin(feature))/diff
        #standardized_test_feature = (tx_test_normalized.T[index]-feature.mean())/feature.std()
        #standardized_feature = (feature-feature.mean())/feature.std()
        if ((diff != 0) & (diff != 1)) : #avoids normalizing categorical features resulting in singular matrices
            feature[:] = normalized_feature
            tx_test_normalized.T[index, :] = normalized_test_feature
    return tx_normalized, tx_test_normalized

''' Calculate accuracy of model'''
def accuracy_function(weights, x, y) :
    y_pred = predict_labels(weights, x)
    return np.count_nonzero(y == y_pred) / (y.shape[0])

''' Compute the overall accuracy when the dataset has been divided into categories '''
def total_accuracy(weights, x, y, masks) :
    total_count = 0
    for i in range(weights.shape[0]) :
        y_pred = predict_labels(weights[i], x[i])
        total_count += np.count_nonzero(y[masks[i]] == y_pred)
    return total_count / (y.shape[0])


""" Build k indices for k-fold """
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


'''
Doing k-fold cross validation for the parameter(s) entered.
Return the average accuracy obtained for the training and testing set
'''
def cross_validation(k_fold, method, y, tx_poly, lambda_ = 0, max_iters = 1, gamma = 0, cross = False):

    # define lists to store the accuracy of training data and test data for the given parameter
    acc_tr_param = []
    acc_te_param = []

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed = 2)

    initial_w = np.zeros(tx_poly.shape[1])

    for k in range(k_fold):

        te_indice = k_indices[k]
        tr_indice = np.delete(k_indices, k, 0)
        tr_indice = np.ndarray.flatten(tr_indice)

        tr_x = tx_poly[tr_indice]
        tr_y = y[tr_indice]
        te_x = tx_poly[te_indice]
        te_y = y[te_indice]

        # method call
        w, loss = method(tr_y, tr_x, lambda_, initial_w, max_iters, gamma)

        acc_tr_param.append(accuracy_function(w, tr_x, tr_y))
        acc_te_param.append(accuracy_function(w, te_x, te_y))

    return np.mean(acc_tr_param), np.mean(acc_te_param)

'''
Determine the optimal parameter(s) for the given method. Putting default values for the different parameters enable
to call determine_parameter for every method.
method is the function that compute the weights and the loss.
Return the optimal parameters: degree, lambda, gamma (if only a subset of the parameters is used, consider only
the wanted parameters and others are just default values)
'''
def determine_parameter(method, tx, y, cross, degrees = [1], lambdas = [0], gammas = [0], k_fold = 4, max_iters = 1, name = 'x'):

    # define a matrix to store the accuracy of training data and test data
    acc_tr_matrix = np.zeros(shape = (len(degrees), len(lambdas), len(gammas)))
    acc_te_matrix = np.zeros(shape = (len(degrees), len(lambdas), len(gammas)))
    initial_w = np.zeros(tx.shape[1])
    for h, degree in enumerate(degrees) :
        # feature expansion
        if cross:
            tx_poly = build_all(tx, degree)
        else:
            tx_poly = build_multi_poly(tx, degree)
        for i, lambda_ in enumerate(lambdas) :
            for j, gamma in enumerate(gammas) :
                acc_tr_matrix[h, i, j], acc_te_matrix[h, i, j] = cross_validation(k_fold, method, y, tx_poly, lambda_ = lambda_, max_iters = max_iters, gamma = gamma, cross = cross)

    # find the indices of the maximum accuracy in 'acc_te_matrix'
    max_acc_index = np.unravel_index(acc_te_matrix.argmax(), acc_te_matrix.shape)
     # maximum accuracy in 'acc_te_matrix'
    max_acc = acc_te_matrix[max_acc_index]

    # optimal parameters
    degree = degrees[max_acc_index[0]]
    lambda_ = lambdas[max_acc_index[1]]
    gamma = gammas[max_acc_index[2]]

    # Plot only the wanted parameter(s) in function of the accuracy.
    # When plotting a specific parameter others parameters are optimal.
    if len(degrees) > 1:
        if (lambda_ != 0) : print("Lambda = ", lambda_)
        if (gamma != 0) : print("Gamma = ", gamma)
        title = 'Optimal degree for ' + name
        degree_plot = plot_train_test(acc_tr_matrix[:, max_acc_index[1], max_acc_index[2]], acc_te_matrix[:, max_acc_index[1], max_acc_index[2]], degrees, title, 'degree')
        print("Optimal degree :", degree, '\n')
    if len(lambdas) > 1:
        if (degree != 1) : print("Degree = ", degree)
        if (gamma != 0) : print("Gamma = ", gamma)
        title = 'Optimal lambda for ' + name
        lambdas_plot = plot_train_test(acc_tr_matrix[max_acc_index[0], :, max_acc_index[2]], acc_te_matrix[max_acc_index[0], :, max_acc_index[2]], lambdas, title, 'lambda', log = True)
        print("Optimal lambda :", lambda_, '\n')
    if len(gammas) > 1:
        if (degree != 1) : print("Degree = ", degree)
        if (lambda_ != 0) : print("Lambda = ", lambda_)
        title = 'Optimal gamma for ' + name
        gammas_plot = plot_train_test(acc_tr_matrix[max_acc_index[0], max_acc_index[1], :], acc_te_matrix[max_acc_index[0], max_acc_index[1], :], gammas, title, 'gamma', log = True)
        print("Optimal gamma :", gamma, '\n')
    print("Max accuracy :", max_acc)

    return degree, lambda_, gamma

'''
Generate the optimal parameters that enable to train the model into the whole dataset and to have the optimal
weights.
X: categorized dataset(X[0] represent the first category)

Return all useful information of the model: optimal weights for each category, dataset
matrix with feature expansion for each category, optimal parameters and the accuracy.
'''
def analyze_method(method, X, y, masks, degrees = [1], lambdas = [0], gammas = [0], max_iters = 1, k_fold = 4, name = "x", cross = False) :
    optimal_degree = np.zeros(X.shape[0], dtype = int)
    optimal_lambda = np.zeros(X.shape[0])
    optimal_gamma = np.zeros(X.shape[0])
    # expansion features of the dataset with the optimal degree.
    X_poly = []
    # weights
    W = []
    # loss
    L = []

    # for each catagory fill in W and L.
    for i in range(X.shape[0]) :
        temp_name = name + str(i+1)
        optimal_degree[i], optimal_lambda[i], optimal_gamma[i] = determine_parameter(method, X[i], y[masks[i]], cross, degrees = degrees, lambdas = lambdas, gammas = gammas, max_iters = max_iters, k_fold = k_fold, name = temp_name)
        X_poly.append(build_multi_poly(X[i], optimal_degree[i]))
        initial_w = np.zeros(X_poly[i].shape[1])
        w_temp, l_temp = method(y[masks[i]], X_poly[i], optimal_lambda[i], initial_w, max_iters, optimal_gamma[i])
        W.append(w_temp)
        L.append(l_temp)
    X_poly = np.array([X_poly[0], X_poly[1], X_poly[2], X_poly[3]])
    W = np.array([W[0], W[1], W[2], W[3]])
    L = np.array([L[0], L[1], L[2], L[3]])


    accuracy = total_accuracy(W, X_poly, y, masks)
    print("Accuracy with optimal parameters is :", accuracy * 100, '%')

    return W, X_poly, optimal_degree, optimal_lambda, optimal_gamma, accuracy
