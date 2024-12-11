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

import numpy as np
from copy import deepcopy
import os

class MyDecisionTreeClassifier:
    """
    Represents a decision tree classifier built using TDIDT.
    """

    def __init__(self):
        """
        Initializes the decision tree classifier with default attributes.
        """
        self.X_train = None
        self.y_train = None
        self.header = None
        self.attribute_domains = None
        self.tree = None

    def fit(self, X_train, y_train):
        """
        Fits a decision tree classifier to the given training data using the TDIDT algorithm.

        Parameters:
        X_train (list of list): Training data features.
        y_train (list): Training data labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.header = ["att" + str(i) for i in range(len(X_train[0]))]

        # Determine attribute domains and sort them to ensure deterministic splits
        self.attribute_domains = {}
        for i, att in enumerate(self.header):
            col = self._get_column(X_train, i)
            unique_vals = list(set(col))
            unique_vals.sort()  # Ensure deterministic ordering
            self.attribute_domains[att] = unique_vals

        instance_indices = list(range(len(y_train)))
        available_attributes = deepcopy(self.header)
        self.tree = self._tdidt(instance_indices, available_attributes)


    def predict(self, X_test):
        """
        Predicts the labels for the given test data.

        Parameters:
        X_test (list of list): Test data features.

        Returns:
        list: Predicted labels for the test data.
        """
        preds = []
        for val in X_test:
            prediction = self._tdidt_predict(self.tree, val)
            preds.append(prediction)
        return preds

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """
        Prints the decision rules derived from the decision tree.

        Parameters:
        attribute_names (list, optional): Names of the attributes for better readability. Defaults to None.
        class_name (str, optional): Name of the class label in the rules. Defaults to "class".
        """
        if attribute_names is None:
            attribute_names = self.header

        rules = ""
        tree = deepcopy(self.tree)
        # The top of the tree should be ["Attribute", attX, ...]
        if tree[0] == "Attribute":
            root_att_name = attribute_names[self.header.index(tree[1])]
            for i in range(2, len(tree)):
                rules += f"IF {root_att_name} == {tree[i][1]} "
                rules += self._print_decision_rules_rec(tree[i][2], attribute_names, class_name)
        print(rules)

    def _print_decision_rules_rec(self, tree, attribute_names, class_name):
        """
        Recursively build decision rules from the tree.
        """
        node_type = tree[0]
        if node_type == "Leaf":
            return f"THEN {class_name} = {tree[1]}\n"
        elif node_type == "Attribute":
            rules = ""
            att_name = attribute_names[self.header.index(tree[1])]
            for i in range(2, len(tree)):
                val = tree[i][1]
                rules += f"AND {att_name} == {val} "
                rules += self._print_decision_rules_rec(tree[i][2], attribute_names, class_name)
            return rules
        else:
            return ""

    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """
        Visualizes the decision tree using Graphviz.

        Parameters:
        dot_fname (str): Filename for the Graphviz .dot file.
        pdf_fname (str): Filename for the output PDF file.
        attribute_names (list, optional): Names of the attributes for better readability. Defaults to None.
        """
        if attribute_names is None:
            attribute_names = self.header

        self.part_count = 0
        with open(dot_fname, "w") as outfile:
            outfile.write("graph g {\n")
            self._visualize_tree_rec(self.tree, outfile, attribute_names)
            outfile.write("}\n")

        os.system(f"dot -Tpdf {dot_fname} -o {pdf_fname}")

    def _visualize_tree_rec(self, tree, outfile, attribute_names):
        """
        Prints the decision rules derived from the decision tree.

        Parameters:
        attribute_names (list, optional): Names of the attributes for better readability. Defaults to None.
        class_name (str, optional): Name of the class label in the rules. Defaults to "class".
        """
        node_type = tree[0]
        if node_type == "Attribute":
            att_idx = self.header.index(tree[1])
            label = attribute_names[att_idx]
            outfile.write(f"node{self.part_count}[shape=box, label=\"{label}\"];\n")
            node_id = self.part_count
            self.part_count += 1
            # For each value branch
            for i in range(2, len(tree)):
                val = tree[i][1]
                outfile.write(f"node{node_id}--node{self.part_count}[label=\"{val}\"];\n")
                child_id = self.part_count
                self.part_count += 1
                self._visualize_tree_child(tree[i][2], outfile, attribute_names, child_id)
        elif node_type == "Leaf":
            outfile.write(f"node{self.part_count}[shape=oval, label=\"{tree[1]}\"];\n")
            self.part_count += 1

    def _visualize_tree_child(self, subtree, outfile, attribute_names, node_id):
        """
        Visualizes the decision tree using Graphviz.

        Parameters:
        dot_fname (str): Filename for the Graphviz .dot file.
        pdf_fname (str): Filename for the output PDF file.
        attribute_names (list, optional): Names of the attributes for better readability. Defaults to None.
        """
        node_type = subtree[0]
        if node_type == "Attribute":
            att_idx = self.header.index(subtree[1])
            label = attribute_names[att_idx]
            outfile.write(f"node{node_id}[shape=box, label=\"{label}\"];\n")
            parent_id = node_id
            for i in range(2, len(subtree)):
                val = subtree[i][1]
                outfile.write(f"node{parent_id}--node{self.part_count}[label=\"{val}\"];\n")
                child_id = self.part_count
                self.part_count += 1
                self._visualize_tree_child(subtree[i][2], outfile, attribute_names, child_id)
        elif node_type == "Leaf":
            outfile.write(f"node{node_id}[shape=oval, label=\"{subtree[1]}\"];\n")

    def _tdidt(self, instance_indices, available_attributes):
        """
        Recursively builds the decision tree using the TDIDT algorithm.

        Parameters:
        instance_indices (list): Indices of the current instances.
        available_attributes (list): List of attributes available for splitting.

        Returns:
        list: The decision tree structure.
        """
        split_att = self._split_attribute(instance_indices, available_attributes)
        available_attributes = [a for a in available_attributes if a != split_att]
        tree = ["Attribute", split_att]

        partitions = self._partition_instances(instance_indices, split_att)

        current_size = len(instance_indices)
        for value, subset_indices in partitions.items():
            val_subtree = ["Value", value]
            if len(subset_indices) > 0 and self._all_same_class(subset_indices):
                leaf_val = self.y_train[subset_indices[0]]
                leaf = ["Leaf", leaf_val, len(subset_indices), current_size]
                val_subtree.append(leaf)
                tree.append(val_subtree)

            elif len(subset_indices) > 0 and len(available_attributes) == 0:
                # Majority class
                vals, freqs = self._get_frequencies([self.y_train[i] for i in subset_indices])
                majority_val = vals[freqs.index(max(freqs))]
                leaf = ["Leaf", majority_val, max(freqs), current_size]
                val_subtree.append(leaf)
                tree.append(val_subtree)

            elif len(subset_indices) == 0:
                # Empty partition - majority of parent's subset
                vals, freqs = self._get_frequencies([self.y_train[i] for i in instance_indices])
                majority_val = vals[freqs.index(max(freqs))]
                leaf = ["Leaf", majority_val, max(freqs), current_size]
                val_subtree.append(leaf)
                tree.append(val_subtree)
            else:
                subtree = self._tdidt(subset_indices, deepcopy(available_attributes))
                val_subtree.append(subtree)
                tree.append(val_subtree)

        return tree

    def _tdidt_predict(self, tree, instance):
        """
        Makes a prediction for a single instance using the decision tree.

        Parameters:
        tree (list): The decision tree structure.
        instance (list): Feature values of the instance.

        Returns:
        Any: The predicted class label.
        """
        node_type = tree[0]
        if node_type == "Leaf":
            return tree[1]
        elif node_type == "Attribute":
            att_name = tree[1]
            att_idx = self.header.index(att_name)
            instance_val = instance[att_idx]
            for i in range(2, len(tree)):
                if tree[i][1] == instance_val:
                    return self._tdidt_predict(tree[i][2], instance)
            # If no branch matches, return None or majority
            return None

    def _partition_instances(self, instance_indices, att):
        """
        Partitions the instances based on the given attribute.

        Parameters:
        instance_indices (list): Indices of the current instances.
        att (str): Attribute name to partition on.

        Returns:
        dict: A dictionary mapping attribute values to subsets of indices.
        """
        att_idx = self.header.index(att)
        partitions = {}
        # Use the sorted domain we established
        for val in self.attribute_domains[att]:
            partitions[val] = []
        for i in instance_indices:
            val = self.X_train[i][att_idx]
            partitions[val].append(i)
        return partitions

    def _split_attribute(self, instance_indices, available_attributes):
        """
        Determines the best attribute to split the instances on based on entropy.

        Parameters:
        instance_indices (list): Indices of the current instances.
        available_attributes (list): List of available attributes for splitting.

        Returns:
        str: The attribute with the lowest entropy.
        """
        att_entropies = []
        for att in available_attributes:
            ent = self._attribute_entropy(instance_indices, att)
            att_entropies.append(ent)
        # Choose attribute with smallest entropy
        min_entropy_idx = att_entropies.index(min(att_entropies))
        return available_attributes[min_entropy_idx]

    def _attribute_entropy(self, instance_indices, att):
        """
        Calculates the entropy of an attribute.

        Parameters:
        instance_indices (list): Indices of the current instances.
        att (str): Attribute name to calculate entropy for.

        Returns:
        float: The entropy value.
        """
        partitions = self._partition_instances(instance_indices, att)
        total = len(instance_indices)
        entropy = 0.0
        for val, subset in partitions.items():
            if len(subset) == 0:
                continue
            subset_labels = [self.y_train[i] for i in subset]
            vals, freqs = self._get_frequencies(subset_labels)
            subset_entropy = 0.0
            for f in freqs:
                p = f / len(subset)
                if p > 0:
                    subset_entropy -= p * np.log2(p)
            entropy += (len(subset) / total) * subset_entropy
        return entropy

    def _get_column(self, matrix, col_index):
        """
        Extracts a column from a matrix.

        Parameters:
        matrix (list of list): The matrix to extract the column from.
        col_index (int): The column index.

        Returns:
        list: The extracted column.
        """
        return [row[col_index] for row in matrix]

    def _all_same_class(self, instance_indices):
        """
        Checks if all instances belong to the same class.

        Parameters:
        instance_indices (list): Indices of the current instances.

        Returns:
        bool: True if all instances have the same class label, False otherwise.
        """
        if len(instance_indices) == 0:
            return True
        first_label = self.y_train[instance_indices[0]]
        return all(self.y_train[i] == first_label for i in instance_indices)

    def _get_frequencies(self, values):
        """
        Computes the frequency of each unique value in a list.

        Parameters:
        values (list): List of values.

        Returns:
        tuple: A tuple containing the unique values and their frequencies.
        """
        vals = list(set(values))
        freqs = [values.count(v) for v in vals]
        return vals, freqs


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
