"""
Matt S
Ryan St. Mary
"""


from __future__ import division

from classifiers import myutils
import random
import numpy as np
from scipy.stats import mode
from classifiers.myutils import *
from classifiers.myclassifiers import *
import numpy as np
import operator
import random
from copy import deepcopy
import numpy as np
from collections import Counter
import random
from copy import deepcopy
import os

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
                all_distances = [
                    (euclidian_distance(value, train_val), index)
                    for index, train_val in enumerate(self.X_train)
                ]
            else:
                all_distances = [
                    (sum(1 for i in range(len(value)) if value[i] != train[i]), index)
                    for index, train in enumerate(self.X_train)
                ]

            all_distances.sort(key=operator.itemgetter(0))

            dist = [neighbor[0] for neighbor in all_distances[: self.n_neighbors]]
            idx = [neighbor[1] for neighbor in all_distances[: self.n_neighbors]]
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
            freqs = get_frequencies(neighbors_classes)
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
        """Initializer for DummyClassifier."""
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
            freqs = get_frequencies(y_train)
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
    """
    Represents a Naive Bayes classifier for categorical data.

    Attributes:
        priors (dict): The prior probabilities computed for each label in the training set.
            Format: {label: prior_probability}.
        posteriors (dict): The posterior probabilities computed for each
            attribute value/label pair in the training set.
            Format: {attribute_index: {label: {attribute_value: posterior_probability}}}.
        is_fitted (bool): Indicates whether the classifier has been fitted with training data.

    Methods:
        fit(X_train, y_train):
            Fits the classifier to the training data by computing priors and posteriors.
        predict(X_test):
            Makes predictions for new data based on the computed priors and posteriors.

    Notes:
        - Loosely based on sklearn's Naive Bayes classifiers:
          https://scikit-learn.org/stable/modules/naive_bayes.html
        - Terminology:
            - Instance = sample = row
            - Attribute = feature = column
        - This implementation assumes categorical data. For numerical attributes, discretization
          or a Gaussian model would be necessary.

    """

    def __init__(self):
        """
        Initializes the Naive Bayes classifier.

        Attributes:
            - priors: Dictionary to store prior probabilities for each label.
            - posteriors: Nested dictionary to store posterior probabilities.
            - is_fitted: Boolean to indicate whether the model has been trained.
        """
        self.priors = {}  # {label: prior_probability}
        self.posteriors = (
            {}
        )  # {attribute_index: {label: {attribute_value: posterior_probability}}}
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        Fits the Naive Bayes classifier to the training data.

        Args:
            X_train (list of list of obj): A 2D list where each row is a training instance
                and each column is a feature. Shape: (n_train_samples, n_features).
            y_train (list of obj): A list of target labels, parallel to X_train.
                Shape: (n_train_samples).

        Raises:
            ValueError: If X_train or y_train is empty.
            ValueError: If X_train and y_train have mismatched lengths.
            ValueError: If X_train is not a 2D list.
            ValueError: If y_train contains no labels.

        Notes:
            - Computes the prior probabilities for each label.
            - Computes the posterior probabilities for each attribute value/label pair.
            - Handles categorical data by estimating probabilities based on observed frequencies.
        """
        if not X_train or not y_train:
            raise ValueError("Training data (X_train and y_train) cannot be empty.")
        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must have the same number of samples."
            )
        if not all(isinstance(row, list) for row in X_train):
            raise ValueError("X_train must be a 2D list.")
        if len(set(y_train)) == 0:
            raise ValueError("y_train must contain at least one label.")

        total_samples = len(y_train)
        unique_labels = set(y_train)
        self.priors = {
            label: y_train.count(label) / total_samples for label in unique_labels
        }
        n_features = len(X_train[0])
        self.posteriors = {
            i: {label: {} for label in unique_labels} for i in range(n_features)
        }
        for feature_index in range(n_features):
            unique_values = set(row[feature_index] for row in X_train)

            for label in unique_labels:
                label_count = y_train.count(label)

                for value in unique_values:
                    count_value_label = sum(
                        1
                        for i in range(total_samples)
                        if X_train[i][feature_index] == value and y_train[i] == label
                    )
                    self.posteriors[feature_index][label][value] = (
                        count_value_label / label_count if label_count > 0 else 0
                    )
        self.is_fitted = True

    def predict(self, X_test):
        """
        Makes predictions for test instances in X_test.

        Args:
            X_test (list of list of obj): A 2D list where each row is a test instance
                and each column is a feature. Shape: (n_test_samples, n_features).

        Returns:
            list of obj: Predicted labels for each test instance, parallel to X_test.

        Raises:
            RuntimeError: If the classifier has not been fitted.
            ValueError: If X_test is empty or not a 2D list.

        Notes:
            - Predictions are based on the maximum a posteriori probability (MAP) estimate.
            - Handles categorical data only. For numerical data, preprocessing is required.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "The classifier must be fitted before calling predict()."
            )
        if not X_test or not all(isinstance(row, list) for row in X_test):
            raise ValueError("X_test must be a non-empty 2D list.")
        y_predicted = []

        for instance in X_test:
            label_probabilities = {}
            for label in self.priors:
                prob = self.priors[label]
                for feature_index, value in enumerate(instance):
                    prob *= self.posteriors[feature_index][label].get(value, 0)
                label_probabilities[label] = prob
            y_predicted.append(max(label_probabilities, key=label_probabilities.get))

        return y_predicted


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    """

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier."""
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using TDIDT."""
        self.header = ["att" + str(i) for i in range(len(X_train[0]))]
        self.attribute_domains = {}
        for att in self.header:
            col = get_column(X_train, self.header.index(att))
            col = list(set(col))
            self.attribute_domains[att] = col

        train = [X_train[i] + [y_train[i]] for i in range(len(y_train))]
        availible_atts = deepcopy(self.header)
        self.tree = self.tdidt(train, availible_atts)

    def tdidt_predict(self, tree, instance):
        """Uses the tree to make a prediction for a single instance."""
        node_type = tree[0]
        if node_type == "Leaf":
            return tree[1]

        # attribute node case, match instance val to right subtree
        att_idx = self.header.index(tree[1])
        for i in range(2, len(tree)):
            if instance[att_idx] == tree[i][1]:
                return self.tdidt_predict(tree[i][2], instance)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test."""
        preds = []
        for val in X_test:
            prediction = self.tdidt_predict(self.tree, val)
            # Replace None with "Other"
            if prediction is None:
                prediction = "OTHER"
            preds.append(prediction)
        return preds


    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree."""
        rules = ""
        tree = deepcopy(self.tree)

        for i in range(2, len(tree)):
            rules += f"IF {tree[1]} == {tree[i][1]} "
            rules += self.print_decision_rules_rec(tree[i], attribute_names, class_name)

        print(rules)

    def print_decision_rules_rec(self, tree, attribute_names=None, class_name="class"):
        if tree[0] == "Leaf":
            return "THEN " + class_name + " = " + tree[1] + "\n"

        rules = ""
        for i in range(2, len(tree[2])):
            rule = (
                f" AND {tree[2][i]} == {tree[2][i]} "
                + self.print_decision_rules_rec(tree[2][i], attribute_names, class_name)
            )
            rules += rule
        return rules

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree with Graphviz."""
        self.part_count = 0
        outfile = open(dot_fname, "w")
        outfile.write("graph g {")
        self.visualize_tree_rec(self.tree, outfile, attribute_names)
        outfile.write("}")
        os.popen(f"dot -Tpdf -o {pdf_fname} {dot_fname}")

        # TODO: Implement further if required

    def visualize_tree_rec(self, tree, outfile, attribute_names=None):
        if tree[0] == "Attribute":
            if attribute_names:
                outfile.write(
                    f"node{self.part_count}[shape=box, label={attribute_names[attribute_names.index(tree[1])]}];\n"
                )
            else:
                outfile.write(f"node{self.part_count}[shape=box, label={tree[1]}];\n")
            self.part_count += 1

            for i in range(len(tree[2:])):
                self.visualize_tree_rec(tree[i + 2], outfile, attribute_names)

        elif tree[0] == "Value":
            outfile.write(
                f"node{self.part_count-1}--node{self.part_count}[label={tree[1]}]\n"
            )
            self.visualize_tree_rec(tree[2], outfile, attribute_names)

        elif tree[0] == "Leaf":
            outfile.write(f"node{self.part_count}[label={tree[1]}];")
            self.part_count += 1

    def partition_instances(self, current_instances, att):
        """Groups instances by attribute domain."""
        att_idx = self.header.index(att)
        att_domain = self.attribute_domains[att]

        partitions = {}
        for value in att_domain:
            partitions[value] = []
            for instance in current_instances:
                if instance[att_idx] == value:
                    partitions[value].append(instance)
        return partitions

    def split_attribute(self, current_instances, available_attributes):
        """Chooses an attribute to split on using entropy-based selection."""
        att_entropies = []
        for att in available_attributes:
            vals = []
            val_entropies = []
            posteriors = []
            val_col = get_column(current_instances, self.header.index(att))
            for val in self.attribute_domains[att]:
                vals.append(val)
                try:
                    posteriors.append(
                        sum(
                            [
                                1
                                for i in range(len(val_col))
                                if val_col[i] == val
                                and current_instances[i][-1] == "True"
                            ]
                        )
                        / sum([1 for i in val_col if i == val])
                    )
                except:
                    posteriors.append(0)

                # Entropy calculation
                if posteriors[-1] not in [0, 1]:
                    val_entropies.append(
                        0
                        - posteriors[-1] * np.log2(posteriors[-1])
                        - (1 - posteriors[-1]) * np.log2(1 - posteriors[-1])
                    )
                else:
                    val_entropies.append(0)

            avg = 0
            for i in range(len(vals)):
                avg += (
                    val_col.count(vals[i]) / len(current_instances)
                ) * val_entropies[i]
            att_entropies.append(avg)

        # return attribute with smallest entropy
        return available_attributes[att_entropies.index(min(att_entropies))]

    def tdidt(self, current_instances, available_attributes):
        """Constructs the decision tree using TDIDT."""
        split_att = self.split_attribute(current_instances, available_attributes)
        available_attributes.remove(split_att)
        tree = ["Attribute", split_att]

        partitions = self.partition_instances(current_instances, split_att)

        for value in partitions.keys():
            att_partition = partitions[value]
            val_subtree = ["Value", value]

            # Case 1
            if len(att_partition) > 1 and all_same_class(att_partition):
                leaf = [
                    "Leaf",
                    att_partition[0][-1],
                    get_column(current_instances, self.header.index(split_att)).count(
                        value
                    ),
                    len(current_instances),
                ]
                val_subtree.append(leaf)
                tree.append(val_subtree)

            # Case 2 (clash)
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                vals, freqs = get_frequencies(get_column(current_instances, -1))
                val = vals[freqs.index(max(freqs))]
                leaf = ["Leaf", val, max(freqs), len(current_instances)]
                val_subtree.append(leaf)
                tree.append(val_subtree)

            # Case 3 (empty partition)
            elif len(att_partition) == 0:
                vals, freqs = get_frequencies(get_column(current_instances, -1))
                val = vals[freqs.index(max(freqs))]
                leaf = ["Leaf", val, max(freqs), len(current_instances)]
                val_subtree.append(leaf)
                tree.append(val_subtree)

            else:
                subtree = self.tdidt(att_partition, available_attributes.copy())
                val_subtree.append(subtree)
                tree.append(val_subtree)

        return tree


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

    def __init__(
        self,
        n_estimators=10,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        bootstrap=True,
    ):
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
        if isinstance(self.max_features, (int, np.integer)):
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
            tree.header = ["att" + str(i) for i in range(len(X[0]))]
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
        predictions = [
            [predictions[j][i] for j in range(len(predictions))]
            for i in range(len(X_test))
        ]

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
