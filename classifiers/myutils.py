
"""
Matt S
Ryan St. Mary
"""
from __future__ import division
from collections import Counter
import random
import numpy as np


import numpy as np

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
