"""
Ryan St. Mary
10-25-2024
CPSC 322 Fall 2024
PA6
This file defines a few helper functions
"""

# from mysklearn import myevaluation
# from mysklearn import myclassifiers
import numpy as np

def get_column(table, index):
    return [row[index] for row in table]



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
