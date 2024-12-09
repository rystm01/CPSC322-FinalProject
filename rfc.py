import numpy as np
from collections import Counter
import random
from copy import deepcopy
from classifiers.myclassifiers import *

class RandomForestClassifier:
    """
    A random forest classifier implementation.

    The Random Forest algorithm is an ensemble method that combines multiple decision trees
    to make a more robust and accurate prediction. It achieves this by training each decision tree
    on a random subset of the data and features and then aggregating the predictions of all trees.

    Attributes:
        n_estimators (int):
            The number of decision trees in the forest.
            Default is 10.

        max_features (str, int, or float):
            The number of features to consider when looking for the best split at each node.
            - If int, `max_features` specifies the exact number of features.
            - If float, `max_features` is a fraction of the total features (e.g., 0.5 means 50% of features).
            - If "sqrt", use the square root of the total number of features.
            - If "log2", use the base-2 logarithm of the total number of features.
            - If None, all features are considered.
            Default is "sqrt".

        max_depth (int):
            The maximum depth of each decision tree. If None, nodes are expanded until all leaves
            are pure or contain fewer than `min_samples_split` samples. Default is None.

        min_samples_split (int):
            The minimum number of samples required to split an internal node. Default is 2.

        bootstrap (bool):
            Whether bootstrap sampling is used when building decision trees.
            - If True, each tree is trained on a bootstrap sample (a random sample with replacement).
            - If False, all trees are trained on the full dataset. Default is True.

        forest (list of MyDecisionTreeClassifier):
            A list of trained decision tree classifiers. Each tree is an instance of `MyDecisionTreeClassifier`.

    Methods:
        __init__(n_estimators, max_features, max_depth, min_samples_split, bootstrap):
            Initializes the RandomForestClassifier with specified parameters.

        fit(X, y):
            Trains the random forest on the provided training dataset.

        predict(X_test):
            Predicts the class for each sample in the test dataset using majority voting.

        score(X_test, y_test):
            Calculates the accuracy of the classifier on the test dataset.
    """

    def __init__(self, n_estimators=10, max_features="sqrt", max_depth=None, min_samples_split=2, bootstrap=True):
        """
        Initializes the RandomForestClassifier with the specified hyperparameters.

        Args:
            n_estimators (int, optional):
                Number of decision trees in the forest. Default is 10.

            max_features (str, int, or float, optional):
                The number of features to consider for splits. Default is "sqrt".

            max_depth (int, optional):
                Maximum depth of each decision tree. If None, trees grow until leaves are pure.
                Default is None.

            min_samples_split (int, optional):
                Minimum number of samples required to split a node. Default is 2.

            bootstrap (bool, optional):
                Whether to use bootstrap sampling when building trees. Default is True.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.forest = []

    def _get_max_features(self, n_features):
        """
        Determines the number of features to use when splitting nodes.

        Args:
            n_features (int):
                Total number of features in the dataset.

        Returns:
            int: Number of features to consider for splits.

        Raises:
            ValueError: If `max_features` has an unrecognized value.
        """
        if isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        elif self.max_features is None:
            return n_features
        else:
            raise ValueError(f"Unknown max_features value: {self.max_features}")

    def fit(self, X, y):
        """
        Trains the random forest by fitting each decision tree on a random subset of the data.

        Args:
            X (list of list of obj):
                Training feature matrix. Each row represents a sample, and each column represents a feature.

            y (list of obj):
                Target values corresponding to the training samples in X.

        Returns:
            None
        """
        n_samples, n_features = len(X), len(X[0])
        max_features = self._get_max_features(n_features)

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_subset = [X[i] for i in indices]
                y_subset = [y[i] for i in indices]
            else:
                X_subset, y_subset = deepcopy(X), deepcopy(y)

            # Train a decision tree
            tree = MyDecisionTreeClassifier()
            tree.header = ['att' + str(i) for i in range(len(X[0]))]
            tree.fit(X_subset, y_subset)
            self.forest.append(tree)

    def predict(self, X_test):
        """
        Predicts the class for each sample in the test dataset using majority voting.

        Args:
            X_test (list of list of obj):
                Test feature matrix. Each row represents a sample.

        Returns:
            list of obj: Predicted class labels for each sample in X_test.
        """
        predictions = []
        for tree in self.forest:
            tree_predictions = tree.predict(X_test)
            predictions.append(tree_predictions)

        # Transpose predictions to organize them per sample
        predictions = [[predictions[j][i] for j in range(len(predictions))] for i in range(len(X_test))]

        # Perform majority voting
        majority_votes = []
        for pred in predictions:
            vote_count = Counter(pred)
            majority_class = max(vote_count, key=vote_count.get)
            majority_votes.append(majority_class)
        
        return majority_votes
    
    def score(self, X_test, y_test):
        """
        Calculates the accuracy of the classifier on the test dataset.

        Args:
            X_test (list of list of obj):
                Test feature matrix. Each row represents a sample.

            y_test (list of obj):
                True class labels for the test samples.

        Returns:
            float: The accuracy of the classifier as a fraction (0.0 to 1.0).
        """
        predictions = self.predict(X_test)
        correct = 0
        total = 0
        for pred, actual in zip(predictions, y_test):
            if pred == actual:
                correct += 1
            total += 1
        
        return correct / total
