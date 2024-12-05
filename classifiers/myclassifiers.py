"""
Ryan St. Mary
10-25-2024
CPSC 322 Fall 2024
PA6
This file defines classifiers
"""


from classifiers import myutils
import numpy as np
import operator

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

