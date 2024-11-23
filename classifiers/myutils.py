"""
Ryan St. Mary
10-25-2024
CPSC 322 Fall 2024
PA6
This file defines a few helper functions
"""

from mysklearn import myevaluation
from mysklearn import myclassifiers
import numpy as np

def get_column(table, index):
    return [row[index] for row in table]



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

        knn_pred = knn.predict(X_test, True)
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

def bootstrap_method(X, y, k=10):
    """
    performs bootstap method

    input:
    X (list of list): X data
    y (list) : y labels
    k (int) : num times to sample

    output:
    knn_acc, knn_err, dummy_acc, dummy_err (floats):
    respective accuracy and error of each model

    """
    knn = myclassifiers.MyKNeighborsClassifier()
    dummy = myclassifiers.MyDummyClassifier()
    num_preds = 0
    knn_acc = 0
    dummy_acc = 0
    for i in range(10):

        X_train, X_test, y_train, y_test = myevaluation.bootstrap_sample(X, y, random_state=i)

         
        knn.fit(X_train, y_train)
        dummy.fit(0, y_train)

        knn_preds = knn.predict(X_test)
        dummy_preds = dummy.predict(X_test)

        num_preds+= len(knn_preds)
        knn_acc += myevaluation.accuracy_score(y_test, knn_preds, normalize=False)
        dummy_acc += myevaluation.accuracy_score(y_test, dummy_preds, normalize=False)
    
    knn_acc /= num_preds
    dummy_acc /= num_preds
    return knn_acc, 1-knn_acc, dummy_acc, 1-dummy_acc

def randomize_in_place(alist, parallel_list=None):
    """
    shuffles a list and shuffles the parrellel list in the
    same way to retain parralel property if given

    input:
    alist (list): list to be shuffles
    parralell_list (list): other list to be shuffled

    notes: shuffles in place, no return values
    """
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]



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
    col_unique = sorted(list(set(col)))
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
