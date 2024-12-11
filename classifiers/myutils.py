
"""
Matt S
Ryan St. Mary
"""
from __future__ import division
from collections import Counter
import random
import classifiers.myevaluation as myevaluation
import classifiers.myclassifiers as myclassifiers
import numpy as np
import math

def get_column(table, index):
    return [row[index] for row in table]

def all_same_class(instances):
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    # get here, then all same class labels
    return True 


def get_frequencies(col):
    """
    gets frequenceis of every unique value from a 1d list

    input: 
    col (list of any): 1d list of values

    output:
    frequencies (list of 2 lists): parallel lists. frequencies[0] is 
        each unqiue value in col, frequencies[1] is the number of times that
        value appears 
    """
    col_unique = list(set(col))
    frequencies = [col_unique, []]

    for val in col_unique:
        frequencies[1].append(col.count(val))

    return frequencies



def euclidian_distance(v1, v2):
    """
    calculates euclidian distance from v1 to v2
    
    input:
    v1 (list of float or int): first point in space
    v2 (list of float or int): second poinr in space

    output:
    euclidian distance (float): distance from v1 to v2

    notes:
    v1 and v2 must be same length/in the same dimension

    """
    return np.sqrt(sum([(v1[i] - v2[i])**2 for i in range(len(v2))]))



def shuffle_in_unison(a, b):
    """ Shuffles two lists of equal length and keeps corresponding elements in the same index. """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def entropy(Y):
    """ In information theory, entropy is a measure of the uncertanty of a random sample from a group. """
    
    distribution = Counter(Y)
    s = 0.0
    total = len(Y)
    for y, num_y in distribution.items():
        probability_y = (num_y/total)
        s += (probability_y)*np.log(probability_y)
    return -s


def information_gain(y, y_true, y_false):
    """ The reduction in entropy from splitting data into two groups. """
    return entropy(y) - (entropy(y_true)*len(y_true) + entropy(y_false)*len(y_false))/len(y)



def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    n_labels = len(labels)
    matrix = [[0] * n_labels for _ in range(n_labels)]
    for t, p in zip(y_true, y_pred):
        true_idx = label_to_index[t]
        pred_idx = label_to_index[p]
        matrix[true_idx][pred_idx] += 1
    return matrix


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X (list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y (list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size (float or int): float for proportion of dataset to be in test set (e.g., 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g., 5 for 5 instances in test set)
        random_state (int): Integer used for seeding a random number generator for reproducible results
        shuffle (bool): Whether or not to randomize the order of the instances before splitting

    Returns:
        X_train (list of list of obj): The list of training samples
        X_test (list of list of obj): The list of testing samples
        y_train (list of obj): The list of target y values for training (parallel to X_train)
        y_test (list of obj): The list of target y values for testing (parallel to X_test)
    """
    # Set seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Input validation
    if len(X) != len(y):
        raise ValueError("The length of X and y must be the same")

    n_samples = len(X)

    # Determine the number of samples in the test set
    if isinstance(test_size, float):
        n_test = math.ceil(n_samples * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise ValueError("test_size must be a float or int")

    if n_test <= 0 or n_test >= n_samples:
        raise ValueError("test_size must be between 0 and the number of samples")

    # Combine X and y to maintain their parallel structure during shuffling
    combined = list(zip(X, y))

    # Shuffle data if required
    if shuffle:
        np.random.shuffle(combined)

    # Unzip combined list back to X and y
    X, y = zip(*combined)

    # Convert X and y back to lists since zip returns tuples
    X = list(X)
    y = list(y)

    # Split the data into train and test sets
    n_train = n_samples - n_test
    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test




def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float or int): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    """
    if not y_true:  # Check if y_true is empty
        return 0.0 if normalize else 0

    correct_count = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    if normalize:
        return correct_count / len(y_true)
    else:
        return correct_count
  



def cross_val_predict(X, y, stratify=False, k=10):
    """
    Runs the knn, Naive Bayes and dummy classifiers on X and y using k fold split

    input: X (list of list): X data
    y (list): list of labels
    stratify (bool): whether or not to use stratified kfold
    k (int): how many folds to use

    output:
    knn_acc, knn_err, dummy_acc, dummy_err (floats):
    respective accuracy and error of each model

    """
    if stratify:
        folds = myevaluation.stratified_kfold_split(X, y, k)
    else:
        folds = myevaluation.kfold_split(X, k)
    
    y_true = []
    all_knn_preds = []
    all_dummy_preds = []
    all_nb_preds = []
    knn = myclassifiers.MyKNeighborsClassifier()
    dummy = myclassifiers.MyDummyClassifier()
    nb = myclassifiers.MyNaiveBayesClassifier()
    knn_acc = 0
    dummy_acc = 0
    nb_acc = 0
    for fold in folds:
        X_train = [X[i] for i in fold[0]]
        y_train = [y[i] for i in fold[0]]
        X_test = [X[i] for i in fold[1]]
        y_test = [y[i] for i in fold[1]]
        knn.fit(X_train, y_train)
        dummy.fit(0, y_train)
        nb.fit(X_train, y_train)

        knn_pred = knn.predict(X_test, categorical=True)
        knn_acc += myevaluation.accuracy_score(y_test, knn_pred, normalize=False)

        dummy_pred = dummy.predict(X_test)
        dummy_acc += myevaluation.accuracy_score(y_test, dummy_pred, normalize=False)

        nb_pred = nb.predict(X_test)
        nb_acc += myevaluation.accuracy_score(y_test, nb_pred, normalize=False)

        y_true += y_test
        all_knn_preds += knn_pred
        all_dummy_preds += dummy_pred
        all_nb_preds += nb_pred


    
    knn_acc /= len(X)
    dummy_acc /= len(X)
    nb_acc /= len(X)
    

    return knn_acc, dummy_acc, nb_acc, y_true, all_knn_preds, all_dummy_preds, all_nb_preds


