"""
Ryan St. Mary
10-25-2024
CPSC 322 Fall 2024
PA6
This file defines classifiers
"""


from __future__ import division
from classifiers import myutils
import numpy as np
import operator
import numpy as np
from scipy.stats import mode
import random
import numpy as np
from scipy.stats import mode
from myutils import *

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test, categorical=False):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        
        for value in X_test:
            # [(distance, index)]
            if not categorical:
                all_distances = [(myutils.euclidian_distance(value, train_val), index) for index, train_val in enumerate(self.X_train)]
            else:
                all_distances = [(sum(1 for i in range(len(value)) if value[i] != train[i]), index) for index, train in enumerate(self.X_train)]
            
            all_distances.sort(key=operator.itemgetter(0))

            dist = [neighbor[0] for neighbor in all_distances[:self.n_neighbors]]
            idx = [neighbor[1] for neighbor in all_distances[:self.n_neighbors]]
            distances.append(dist)
            neighbor_indices.append(idx)
            # print(value)
            # print([self.X_train[i] for i in idx])
            # print(dist)
        return distances, neighbor_indices

    def predict(self, X_test, categorical=False):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        indexes = self.kneighbors(X_test, categorical)[1]
        y_predicted = []

        
        for test in indexes:
            neighbors_classes = [self.y_train[i] for i in test]
            freqs = myutils.get_frequencies(neighbors_classes)
            y_predicted.append(freqs[0][freqs[1].index(max(freqs[1]))])
        
        return y_predicted



class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
        strategy(string): most frequent or stratified. 


    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self, strategy="most frequent"):
        """Initializer for DummyClassifier.

        """
        self.strategy = strategy
        self.most_common_label = None
        
    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        if self.strategy == "most frequent":
            freqs = myutils.get_frequencies(y_train)
            self.most_common_label = freqs[0][freqs[1].index(max(freqs[1]))]
        elif self.strategy == "stratified":
            self.y_train = y_train



    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.strategy == "most frequent":
            return [self.most_common_label for _ in X_test]
        elif self.strategy == "stratified":
            return [np.random.choice(self.y_train) for _ in X_test]


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = {} # label -> prior
        self.posteriors = {} # att_idx - > labels -> -> x_label -> posterior

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = {}
        self.posteriors = {}
        total = len(y_train)
        y_labels = list(set(y_train))
        
        X_labels = []
        for i in range(len(X_train[0])):
            X_labels.append(list(set(myutils.get_column(X_train, i))))
            self.posteriors[i] = {}
            for label in y_labels:
                self.posteriors[i][label] = {}

        for label in y_labels:
            self.priors[label] = y_train.count(label)/total
            
        
        
        """
        posteriers = {0 : {{'yes' : {1 : val, 2 : val}, 'no' : {1 : val, 2 : val}}}}
        """
        
        

        for i in range(len(y_labels)):              # for each unqiue y
            for j in range(len(X_labels)):          # for each attribute in X
                for k in range(len(X_labels[j])):   # for each unique val in each att
                    self.posteriors[j][y_labels[i]][X_labels[j][k]] = sum([1 for l in range(total) if (X_train[l][j] == 
                                 X_labels[j][k] and y_train[l] == y_labels[i])])/(self.priors[y_labels[i]]*total)
                                                                
        
        

        



    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        y_labels = list(self.priors.keys())
        for x in X_test:
            probs = []
            for label in y_labels:
                prob = 1*self.priors[label]
                for i in range(len(x)):
                    prob*=self.posteriors[i][label][x[i]]
                probs.append(prob)
            y_predicted.append(y_labels[probs.index(max(probs))])

        

        return y_predicted




class DecisionTreeClassifier(object):
    """ A decision tree classifier.

    A decision tree is a structure in which each node represents a binary
    conditional decision on a specific feature, each branch represents the
    outcome of the decision, and each leaf node represents a final
    classification.
    """

    def __init__(self, max_features=lambda x: x, max_depth=10,
                 min_samples_split=2):
        """
        Args:
            max_features: A function that controls the number of features to
                randomly consider at each split. The argument will be the number
                of features in the data.
            max_depth: The maximum number of levels the tree can grow downwards
                before forcefully becoming a leaf.
            min_samples_split: The minimum number of samples needed at a node to
                justify a new node split.
        """
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trunk = None

    def fit(self, X, y):
        """ Builds the tree by choosing decision rules for each node based on
        the data. """
        n_features = X.shape[1]
        n_sub_features = int(self.max_features(n_features))
        feature_indices = random.sample(range(n_features), n_sub_features)
        
        self.trunk = self.build_tree(X, y, feature_indices, 0)

    def predict(self, X):
        """ Predict the class of each sample in X. """
        num_samples = X.shape[0]
        y_pred = np.empty(num_samples, dtype=int)
        
        for j in range(num_samples):
            node = self.trunk
            while isinstance(node, Node):
                if X[j][node.feature_index] <= node.threshold:
                    node = node.branch_true
                else:
                    node = node.branch_false
            y_pred[j] = node

        return y_pred

    def build_tree(self, X, y, feature_indices, depth):
        """ Recursively builds a decision tree. """
        if depth == self.max_depth or len(y) < self.min_samples_split or entropy(y) == 0:
            return mode(y)[0][0]
        
        feature_index, threshold = find_split(X, y, feature_indices)
        X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold)

        if len(y_true) == 0 or len(y_false) == 0:
            return mode(y)[0][0]

        branch_true = self.build_tree(X_true, y_true, feature_indices, depth + 1)
        branch_false = self.build_tree(X_false, y_false, feature_indices, depth + 1)

        return Node(feature_index, threshold, branch_true, branch_false)

def find_split(X, y, feature_indices):
    """ Returns the best split rule for a tree node. """
    best_gain = 0
    best_feature_index = 0
    best_threshold = 0

    for feature_index in feature_indices:
        values = sorted(set(X[:, feature_index])) 

        for j in range(len(values) - 1): 
            threshold = (values[j] + values[j + 1]) / 2 
            X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold)
            gain = information_gain(y, y_true, y_false)

            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold

class Node(object):
    """ A node in a decision tree with the binary condition xi <= t. """
    def __init__(self, feature_index, threshold, branch_true, branch_false):
        self.feature_index = feature_index
        self.threshold = threshold
        self.branch_true = branch_true
        self.branch_false = branch_false

def split(X, y, feature_index, threshold):
    """ Splits X and y based on the binary condition xi <= threshold. """
    mask_true = X[:, feature_index] <= threshold
    mask_false = ~mask_true

    X_true = X[mask_true]
    y_true = y[mask_true]
    X_false = X[mask_false]
    y_false = y[mask_false]

    return X_true, y_true, X_false, y_false


class RandomForestClassifier(object):
    """ A random forest classifier.

    A random forest is a collection of decision trees that vote on a
    classification decision. Each tree is trained with a subset of the data and
    features.
    """

    def __init__(self, n_estimators=32, max_features=np.sqrt, max_depth=10,
                 min_samples_split=2, bootstrap=0.9):
        """
        Args:
            n_estimators: The number of decision trees in the forest.
            max_features: Controls the number of features to randomly consider
                at each split.
            max_depth: The maximum number of levels that the tree can grow
                downwards before forcefully becoming a leaf.
            min_samples_split: The minimum number of samples needed at a node to
                justify a new node split.
            bootstrap: The fraction of randomly chosen data to fit each tree on.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.forest = []

    def fit(self, X, y):
        """ Creates a forest of decision trees using a random subset of data and
            features. """
        self.forest = []
        n_samples = len(y)
        n_sub_samples = round(n_samples * self.bootstrap)
        
        for i in range(self.n_estimators):
            shuffle_in_unison(X, y)
            X_subset = X[:n_sub_samples]
            y_subset = y[:n_sub_samples]

            tree = DecisionTreeClassifier(self.max_features, self.max_depth,
                                          self.min_samples_split)
            tree.fit(X_subset, y_subset)
            self.forest.append(tree)

    def predict(self, X):
        """ Predict the class of each sample in X. """
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict(X)

        # Get the most common prediction
        return mode(predictions, axis=0)[0][0] 

    def score(self, X, y):
        """ Return the accuracy of the prediction of X compared to y. """
        y_predict = self.predict(X)
        n_samples = len(y)
        correct = np.sum(y_predict == y)
        accuracy = correct / n_samples
        return accuracy
