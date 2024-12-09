"""
Matt S
Ryan St. Mary
"""



from __future__ import division
import random
import numpy as np
from scipy.stats import mode
from classifiers.myutils import *
from classifiers.myclassifiers import *
import numpy as np
import operator
import random
from copy import deepcopy

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
                all_distances = [(euclidian_distance(value, train_val), index) for index, train_val in enumerate(self.X_train)]
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
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict): The prior probabilities computed for each label in the training set.
        posteriors(dict): The posterior probabilities computed for each attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier."""
        self.priors = {}
        self.posteriors = {}

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
        from collections import defaultdict

        total_samples = len(y_train)
        label_counts = defaultdict(int)

        # Compute priors
        for label in y_train:
            label_counts[label] += 1
        for label, count in label_counts.items():
            self.priors[label] = count / total_samples

        # Initialize posteriors
        self.posteriors = {label: [] for label in label_counts}

        # Organize data by class
        data_by_class = {label: [] for label in label_counts}
        for x, label in zip(X_train, y_train):
            data_by_class[label].append(x)

        # Compute posteriors for each class
        for label, instances in data_by_class.items():
            attribute_counts = [defaultdict(int) for _ in range(len(X_train[0]))]
            for instance in instances:
                for i, value in enumerate(instance):
                    attribute_counts[i][value] += 1
            total_instances = len(instances)
            for counts in attribute_counts:
                probs = {
                    value: count / total_instances for value, count in counts.items()
                }
                self.posteriors[label].append(probs)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            label_probabilities = {}
            for label in self.priors:
                probability = self.priors[label]
                for i, value in enumerate(instance):
                    probs = self.posteriors[label][i]
                    probability *= probs.get(value, 0)
                label_probabilities[label] = probability
            # Handle the case where all probabilities are zero
            if all(prob == 0 for prob in label_probabilities.values()):
                best_label = max(self.priors, key=self.priors.get)
            else:
                best_label = max(label_probabilities, key=label_probabilities.get)
            y_predicted.append(best_label)
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
        self.header = ['att' + str(i) for i in range(len(X_train[0]))]
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
            preds.append(self.tdidt_predict(self.tree, val))
        return preds

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree."""
        rules = ''
        tree = deepcopy(self.tree)
        
        for i in range(2, len(tree)):
            rules += f'IF {tree[1]} == {tree[i][1]} '
            rules += self.print_decision_rules_rec(tree[i], attribute_names, class_name)
            
        print(rules)

    def print_decision_rules_rec(self, tree, attribute_names=None, class_name="class"):
        if tree[0] == 'Leaf':
            return 'THEN ' + class_name + " = " + tree[1] + "\n"
       
        rules = ''
        for i in range(2, len(tree[2])):
            rule = f' AND {tree[2][i]} == {tree[2][i]} ' + self.print_decision_rules_rec(tree[2][i], attribute_names, class_name)
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
                outfile.write(f"node{self.part_count}[shape=box, label={attribute_names[attribute_names.index(tree[1])]}];\n")
            else:
                outfile.write(f"node{self.part_count}[shape=box, label={tree[1]}];\n")
            self.part_count += 1
            
            for i in range(len(tree[2:])):
                self.visualize_tree_rec(tree[i+2], outfile, attribute_names)

        elif tree[0] == "Value":
            outfile.write(f"node{self.part_count-1}--node{self.part_count}[label={tree[1]}]\n")
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
                        sum([1 for i in range(len(val_col))
                             if val_col[i] == val and current_instances[i][-1] == 'True']) 
                        / sum([1 for i in val_col if i == val])
                    )
                except:
                    posteriors.append(0)
                
                # Entropy calculation
                if posteriors[-1] not in [0, 1]:
                    val_entropies.append(0 - posteriors[-1]*np.log2(posteriors[-1]) 
                                         - (1-posteriors[-1])*np.log2(1-posteriors[-1]))
                else:
                    val_entropies.append(0)
            
            avg = 0
            for i in range(len(vals)):
                avg += (val_col.count(vals[i])/len(current_instances)) * val_entropies[i]
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
                leaf = ["Leaf", att_partition[0][-1],
                        get_column(current_instances, self.header.index(split_att)).count(value),
                        len(current_instances)]
                val_subtree.append(leaf)
                tree.append(val_subtree)

            # Case 2 (clash)
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                vals, freqs = get_frequencies(get_column(current_instances, -1))
                val = vals[freqs.index(max(freqs))]
                leaf = ['Leaf', val, max(freqs), len(current_instances)]
                val_subtree.append(leaf)
                tree.append(val_subtree)

            # Case 3 (empty partition)
            elif len(att_partition) == 0:
                vals, freqs = get_frequencies(get_column(current_instances, -1))
                val = vals[freqs.index(max(freqs))]
                leaf = ['Leaf', val, max(freqs), len(current_instances)]
                # Replace subtree with a leaf
                val_subtree = leaf
                tree.append(val_subtree)

            else:
                subtree = self.tdidt(att_partition, available_attributes.copy())
                val_subtree.append(subtree)
                tree.append(val_subtree)
        
        return tree
    


##Adjusting and improving RadnomForestClassifer locally
